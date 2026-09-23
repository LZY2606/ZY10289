# -*- coding: utf-8 -*-
"""Endpoint coverage driven by the protocol-aware mutation corpus.

Each test class serves a known-good baseline request through a real
oauthlib endpoint (no HTTP is performed) and then replays the corpus
mutations, checking the recorded parser view, validator calls and wire
response against the expected relation (same / reject /
protocol-specific) declared in the mutation table.
"""
import json
import unicodedata
from types import SimpleNamespace
from unittest import mock

from oauthlib.oauth1 import RequestValidator as OAuth1RequestValidator
from oauthlib.oauth1.rfc5849 import Client
from oauthlib.oauth1.rfc5849.endpoints import SignatureOnlyEndpoint
from oauthlib.oauth2.rfc6749 import tokens
from oauthlib.oauth2.rfc6749.endpoints.authorization import (
    AuthorizationEndpoint,
)
from oauthlib.oauth2.rfc6749.endpoints.introspect import IntrospectEndpoint
from oauthlib.oauth2.rfc6749.endpoints.revocation import RevocationEndpoint
from oauthlib.oauth2.rfc6749.endpoints.token import TokenEndpoint
from oauthlib.oauth2.rfc6749.grant_types import AuthorizationCodeGrant
from oauthlib.oauth2.rfc8628.endpoints.device_authorization import (
    DeviceAuthorizationEndpoint,
)
from oauthlib.openid.connect.core.grant_types.authorization_code import (
    AuthorizationCodeGrant as OIDCAuthorizationCodeGrant,
)

from tests.mutation import corpus
from tests.mutation.corpus import (
    PROTOCOL, REJECT, SAME, Mutation, WireRequest,
)
from tests.unittest import TestCase

FORM_HEADERS = {'Content-Type': 'application/x-www-form-urlencoded'}


# --------------------------------------------------------------------
# Endpoint fixtures and baselines
# --------------------------------------------------------------------

def token_baseline():
    return WireRequest(
        'POST', 'https://example.com/token',
        body=('grant_type=authorization_code&code=abc'
              '&redirect_uri=https%3A%2F%2Fback.to%2Fme'
              '&scope=all+of+them&state=caf%C3%A9'),
        headers=dict(FORM_HEADERS))


def make_token_endpoint():
    validator = mock.MagicMock()

    def set_client(request):
        request.client = SimpleNamespace(client_id='me')
        request.user = 'resource-owner'
        return True

    validator.authenticate_client.side_effect = set_client
    validator.get_code_challenge.return_value = None
    validator.is_pkce_required.return_value = False
    grant = AuthorizationCodeGrant(request_validator=validator)
    bearer = tokens.BearerToken(
        validator,
        token_generator=lambda request: 'fixed-access-token',
        refresh_token_generator=lambda request: 'fixed-refresh-token',
        expires_in=1800)
    endpoint = TokenEndpoint(
        'authorization_code', default_token_type=bearer,
        grant_types={'authorization_code': grant})
    return endpoint, validator


def authorization_baseline():
    return WireRequest(
        'GET',
        'https://example.com/authorize?response_type=code&client_id=me'
        '&redirect_uri=https%3A%2F%2Fback.to%2Fme&state=caf%C3%A9'
        '&scope=all+of+them')


def make_authorization_endpoint(oidc=False):
    validator = mock.MagicMock()
    validator.get_code_challenge.return_value = None
    validator.is_pkce_required.return_value = False
    validator.is_origin_allowed.return_value = False
    validator.get_default_redirect_uri.return_value = 'https://back.to/me'
    grant_cls = OIDCAuthorizationCodeGrant if oidc else AuthorizationCodeGrant
    grant = grant_cls(request_validator=validator)
    bearer = tokens.BearerToken(
        validator, token_generator=lambda request: 'fixed-access-token',
        expires_in=1800)
    endpoint = AuthorizationEndpoint(
        'code', default_token_type=bearer, response_types={'code': grant})
    return endpoint, validator


def revocation_baseline():
    return WireRequest(
        'POST', 'https://example.com/revoke',
        body='token=f%C3%B6o+bar&token_type_hint=access_token',
        headers=dict(FORM_HEADERS))


def make_revocation_endpoint():
    validator = mock.MagicMock()
    validator.client_authentication_required.return_value = True
    validator.authenticate_client.return_value = True
    validator.revoke_token.return_value = True
    return RevocationEndpoint(validator), validator


def introspection_baseline():
    return WireRequest(
        'POST', 'https://example.com/introspect',
        body='token=f%C3%B6o+bar&token_type_hint=access_token',
        headers=dict(FORM_HEADERS))


def make_introspect_endpoint():
    validator = mock.MagicMock()
    validator.client_authentication_required.return_value = True
    validator.authenticate_client.return_value = True
    validator.introspect_token.return_value = {'sub': '123'}
    return IntrospectEndpoint(validator), validator


def device_baseline():
    return WireRequest(
        'POST', 'https://example.com/device-authorize',
        body='client_id=f%C3%B6o&scope=bar+baz',
        headers=dict(FORM_HEADERS))


def make_device_endpoint():
    validator = mock.MagicMock()
    validator.validate_client_id.return_value = True
    validator.client_authentication_required.return_value = False
    validator.authenticate_client_id.return_value = True
    endpoint = DeviceAuthorizationEndpoint(
        validator, verification_uri='https://example.com/device',
        user_code_generator=lambda: 'USER-CODE')
    return endpoint, validator


def oidc_baseline():
    return WireRequest(
        'GET',
        'https://example.com/authorize?response_type=code&client_id=me'
        '&redirect_uri=https%3A%2F%2Fback.to%2Fme&scope=openid+profile'
        '&state=caf%C3%A9&nonce=n-0S6_WzA2Mj')


def make_oauth1_endpoint():
    validator = mock.MagicMock(wraps=OAuth1RequestValidator())
    validator.check_client_key.return_value = True
    validator.allowed_signature_methods = ['HMAC-SHA1']
    validator.get_client_secret.return_value = 'bar'
    validator.timestamp_lifetime = 600
    validator.validate_client_key.return_value = True
    validator.validate_timestamp_and_nonce.return_value = True
    validator.dummy_client = 'dummy'
    validator.dummy_secret = 'dummy'
    return SignatureOnlyEndpoint(validator), validator


def oauth1_baseline():
    client = Client('foo', client_secret='bar')
    uri, headers, body = client.sign(
        'https://i.b/protected_resource?foo=b%20r%C3%A9&a=1&b=2')
    return WireRequest('GET', uri, body=body or None, headers=headers)


# --------------------------------------------------------------------
# Shared corpus driver
# --------------------------------------------------------------------

class MutationCorpusBase:
    tracked_params = ()
    protected_methods = ()
    mutations = ()
    baseline = None

    def run_endpoint(self, wire):
        raise NotImplementedError

    def observe(self, wire):
        return corpus.observe(self.run_endpoint, wire, self.validator,
                              self.tracked_params)

    def test_corpus_table_well_formed(self):
        names = [m.name for m in self.mutations]
        self.assertTrue(names, 'corpus table must not be empty')
        self.assertEqual(len(names), len(set(names)),
                         'mutation names must be unique')

    def test_baseline_succeeds(self):
        obs = self.observe(self.baseline)
        self.assertIsNone(obs.exception)
        self.assertFalse(corpus.response_is_error(obs))

    def test_mutations(self):
        baseline = self.observe(self.baseline)
        self.assertIsNone(baseline.exception)
        for mutation in self.mutations:
            with self.subTest(mutation=mutation.name):
                wire = mutation.apply(self.baseline)
                self.assertNotEqual(
                    (wire.method, wire.uri, wire.body,
                     sorted(wire.headers.items())),
                    (self.baseline.method, self.baseline.uri,
                     self.baseline.body,
                     sorted(self.baseline.headers.items())),
                    'mutation %s is a no-op' % mutation.name)
                obs = self.observe(wire)
                if mutation.relation == SAME:
                    corpus.assert_same(self, baseline, obs)
                elif mutation.relation == REJECT:
                    corpus.assert_reject(self, obs, self.protected_methods)
                else:
                    mutation.check(self, baseline, obs)


# --------------------------------------------------------------------
# Protocol-specific checks
# --------------------------------------------------------------------

def _validator_arg_values(obs, method, position):
    return [args[position] for name, args, _ in obs.validator_calls
            if name == method]


def check_duplicate_code_last_wins(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 200)
    t.assertEqual(obs.parser_view['code'], 'xyz')
    t.assertEqual(obs.parser_view['duplicate_params'], ['code'])
    t.assertEqual(_validator_arg_values(obs, 'validate_code', 1), ['xyz'])


def check_tolerated_duplicate(param):
    def check(t, baseline, obs):
        t.assertIsNone(obs.exception)
        t.assertEqual(obs.status, 200)
        t.assertEqual(obs.parser_view['duplicate_params'], [param])
        t.assertEqual(obs.validator_calls, baseline.validator_calls)
    return check


def check_empty_state_inert(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 200)
    t.assertEqual(obs.parser_view['state'], '')
    t.assertEqual(obs.validator_calls, baseline.validator_calls)
    t.assertEqual(corpus.normalized_body(obs.body),
                  corpus.normalized_body(baseline.body))


def check_empty_scope(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 200)
    t.assertEqual(obs.parser_view['scope'], '')
    t.assertEqual(json.loads(obs.body)['scope'], '')


def check_token_encoded_nul_code(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 200)
    t.assertEqual(obs.parser_view['code'], 'abc\x00')
    t.assertEqual(_validator_arg_values(obs, 'validate_code', 1), ['abc\x00'])


def check_unicode_not_normalized(param):
    def check(t, baseline, obs):
        t.assertIsNone(obs.exception)
        t.assertEqual(obs.status, 200)
        expected = unicodedata.normalize('NFD', baseline.parser_view[param])
        t.assertEqual(obs.parser_view[param], expected)
        t.assertNotEqual(obs.parser_view[param], baseline.parser_view[param])
    return check


def check_overlong_code(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 200)
    expected = 'abc' + 'x' * 8192
    t.assertEqual(obs.parser_view['code'], expected)
    t.assertEqual(_validator_arg_values(obs, 'validate_code', 1), [expected])


def check_empty_state_dropped_from_redirect(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 302)
    t.assertEqual(obs.parser_view['state'], '')
    t.assertEqual(obs.headers['Location'], 'https://back.to/me?code=abc')


def check_authorization_encoded_nul_state(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 302)
    t.assertEqual(obs.parser_view['state'], 'café\x00')
    t.assertIn('state=caf%C3%A9%00', obs.headers['Location'])


def check_authorization_unicode_nfd_state(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 302)
    t.assertEqual(obs.parser_view['state'], 'café')
    t.assertIn('state=cafe%CC%81', obs.headers['Location'])


def check_authorization_overlong_state(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 302)
    t.assertEqual(obs.parser_view['state'], 'café' + 'x' * 8192)
    t.assertIn('x' * 100, obs.headers['Location'])


def check_error_redirect(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 302)
    location = obs.headers.get('Location', '')
    t.assertTrue(location.startswith('https://back.to/me?'))
    t.assertIn('error=unsupported_response_type', location)
    t.assertIn('state=caf%C3%A9', location)
    called = [c[0] for c in obs.validator_calls]
    t.assertNotIn('save_authorization_code', called)


def move_client_id_to_post_body(wire):
    moved = corpus.move_to_body('client_id')(wire)
    return WireRequest('POST', moved.uri, moved.body, dict(FORM_HEADERS))


def move_nonce_to_post_body(wire):
    moved = corpus.move_to_body('nonce')(wire)
    return WireRequest('POST', moved.uri, moved.body, dict(FORM_HEADERS))


# --------------------------------------------------------------------
# OAuth 2 token endpoint (authorization_code grant)
# --------------------------------------------------------------------

class TokenEndpointCorpus(MutationCorpusBase, TestCase):
    tracked_params = ('grant_type', 'code', 'redirect_uri', 'scope',
                      'state', 'client_id')
    protected_methods = ('save_token', 'invalidate_authorization_code')

    def setUp(self):
        self.endpoint, self.validator = make_token_endpoint()
        self.baseline = token_baseline()

    def run_endpoint(self, wire):
        return self.endpoint.create_token_response(
            wire.uri, http_method=wire.method, body=wire.body,
            headers=wire.headers)

    mutations = (
        Mutation('percent-encoding-case', SAME, corpus.flip_percent_case),
        Mutation('plus-vs-percent20', SAME, corpus.plus_to_percent20),
        Mutation('parameter-reorder', SAME, corpus.reorder),
        Mutation('duplicate-grant-type', REJECT,
                 corpus.duplicate('grant_type'),
                 note='grant_type is in the enforced duplicate set'),
        Mutation('duplicate-state-tolerated', PROTOCOL,
                 corpus.duplicate('state'),
                 check=check_tolerated_duplicate('state'),
                 note='state is not in the enforced duplicate set'),
        Mutation('duplicate-code-last-value-wins', PROTOCOL,
                 corpus.duplicate('code', 'xyz'),
                 check=check_duplicate_code_last_wins,
                 note='code is not in the enforced duplicate set; the '
                      'parser exposes the last occurrence'),
        Mutation('grant-type-moved-to-query', REJECT,
                 corpus.move_to_query('grant_type'),
                 note='token endpoint rejects query parameters on POST'),
        Mutation('state-moved-to-query', REJECT,
                 corpus.move_to_query('state')),
        Mutation('empty-state', PROTOCOL, corpus.empty_value('state'),
                 check=check_empty_state_inert,
                 note='empty state is kept by the parser but inert here'),
        Mutation('empty-scope', PROTOCOL, corpus.empty_value('scope'),
                 check=check_empty_scope,
                 note='empty scope is not treated as omitted'),
        Mutation('raw-nul-in-code', REJECT, corpus.raw_nul('code'),
                 note='unencoded NUL makes the body unparseable'),
        Mutation('encoded-nul-in-code', PROTOCOL, corpus.encoded_nul('code'),
                 check=check_token_encoded_nul_code,
                 note='%00 reaches the validator verbatim'),
        Mutation('unicode-nfd-state', PROTOCOL, corpus.unicode_nfd('state'),
                 check=check_unicode_not_normalized('state'),
                 note='the framework does not normalize Unicode'),
        Mutation('overlong-code', PROTOCOL, corpus.overlong('code'),
                 check=check_overlong_code,
                 note='no framework length limit; validator decides'),
    )


# --------------------------------------------------------------------
# OAuth 2 authorization endpoint (code grant)
# --------------------------------------------------------------------

class AuthorizationEndpointCorpus(MutationCorpusBase, TestCase):
    tracked_params = ('response_type', 'client_id', 'redirect_uri',
                      'scope', 'state')
    protected_methods = ('save_authorization_code', 'save_token')

    def setUp(self):
        self.endpoint, self.validator = make_authorization_endpoint()
        self.baseline = authorization_baseline()

    def run_endpoint(self, wire):
        with mock.patch('oauthlib.common.generate_token', new=lambda: 'abc'):
            return self.endpoint.create_authorization_response(
                wire.uri, http_method=wire.method, body=wire.body,
                headers=wire.headers, scopes=['all', 'of', 'them'])

    mutations = (
        Mutation('percent-encoding-case', SAME, corpus.flip_percent_case),
        Mutation('plus-vs-percent20', SAME, corpus.plus_to_percent20),
        Mutation('parameter-reorder', SAME, corpus.reorder),
        Mutation('duplicate-scope', REJECT, corpus.duplicate('scope'),
                 note='fatal: duplicate scope is rejected before redirect'),
        Mutation('duplicate-state-conflict', REJECT,
                 corpus.duplicate('state', 'other')),
        Mutation('client-id-moved-to-body', SAME, move_client_id_to_post_body,
                 note='authorization endpoint merges query and body '
                      'parameters'),
        Mutation('empty-redirect-uri', REJECT,
                 corpus.empty_value('redirect_uri'),
                 note='empty redirect_uri is not treated as omitted; '
                      'fatal InvalidRedirectURIError'),
        Mutation('empty-state', PROTOCOL, corpus.empty_value('state'),
                 check=check_empty_state_dropped_from_redirect,
                 note='empty state is dropped from the redirect'),
        Mutation('raw-nul-in-state', REJECT, corpus.raw_nul('state'),
                 note='unencoded NUL makes the query unparseable'),
        Mutation('encoded-nul-in-state', PROTOCOL,
                 corpus.encoded_nul('state'),
                 check=check_authorization_encoded_nul_state,
                 note='%00 is echoed back into the redirect verbatim'),
        Mutation('unicode-nfd-state', PROTOCOL, corpus.unicode_nfd('state'),
                 check=check_authorization_unicode_nfd_state,
                 note='no Unicode normalization on the state echo'),
        Mutation('overlong-state', PROTOCOL, corpus.overlong('state'),
                 check=check_authorization_overlong_state),
        Mutation('error-redirect', PROTOCOL,
                 corpus.set_param('response_type', 'bogus'),
                 check=check_error_redirect,
                 note='normal errors redirect to the registered '
                      'redirect_uri'),
    )


# --------------------------------------------------------------------
# OAuth 2 revocation endpoint
# --------------------------------------------------------------------

def _revoked_tokens(obs):
    return _validator_arg_values(obs, 'revoke_token', 0)


def check_revocation_duplicate_conflict(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 200)
    t.assertEqual(obs.parser_view['token'], 'other')
    t.assertEqual(obs.parser_view['duplicate_params'], ['token'])
    t.assertEqual(_revoked_tokens(obs), ['other'])


def check_revocation_duplicate_tolerated(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 200)
    t.assertEqual(obs.parser_view['duplicate_params'], ['token'])
    t.assertEqual(_revoked_tokens(obs), ['föo bar'])


def check_revocation_empty_hint(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 200)
    t.assertEqual(obs.parser_view['token_type_hint'], '')
    t.assertEqual(_validator_arg_values(obs, 'revoke_token', 1), [''])


def check_revocation_encoded_nul(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 200)
    t.assertEqual(_revoked_tokens(obs), ['föo bar\x00'])


def check_revocation_unicode_nfd(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 200)
    expected = unicodedata.normalize('NFD', 'föo bar')
    t.assertEqual(obs.parser_view['token'], expected)
    t.assertEqual(_revoked_tokens(obs), [expected])


def check_revocation_overlong(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 200)
    t.assertEqual(_revoked_tokens(obs), ['föo bar' + 'x' * 8192])


class RevocationEndpointCorpus(MutationCorpusBase, TestCase):
    tracked_params = ('token', 'token_type_hint')
    protected_methods = ('revoke_token',)

    def setUp(self):
        self.endpoint, self.validator = make_revocation_endpoint()
        self.baseline = revocation_baseline()

    def run_endpoint(self, wire):
        return self.endpoint.create_revocation_response(
            wire.uri, http_method=wire.method, body=wire.body,
            headers=wire.headers)

    mutations = (
        Mutation('percent-encoding-case', SAME, corpus.flip_percent_case),
        Mutation('plus-vs-percent20', SAME, corpus.plus_to_percent20),
        Mutation('parameter-reorder', SAME, corpus.reorder),
        Mutation('duplicate-token-tolerated', PROTOCOL,
                 corpus.duplicate('token'),
                 check=check_revocation_duplicate_tolerated,
                 note='revocation has no duplicate check'),
        Mutation('duplicate-token-last-value-wins', PROTOCOL,
                 corpus.duplicate('token', 'other'),
                 check=check_revocation_duplicate_conflict),
        Mutation('token-moved-to-query', REJECT,
                 corpus.move_to_query('token'),
                 note='revocation rejects query parameters on POST'),
        Mutation('empty-token-type-hint', PROTOCOL,
                 corpus.empty_value('token_type_hint'),
                 check=check_revocation_empty_hint,
                 note='empty hint reaches the validator as empty string'),
        Mutation('raw-nul-in-token', REJECT, corpus.raw_nul('token')),
        Mutation('encoded-nul-in-token', PROTOCOL,
                 corpus.encoded_nul('token'),
                 check=check_revocation_encoded_nul),
        Mutation('unicode-nfd-token', PROTOCOL, corpus.unicode_nfd('token'),
                 check=check_revocation_unicode_nfd),
        Mutation('overlong-token', PROTOCOL, corpus.overlong('token'),
                 check=check_revocation_overlong),
    )


# --------------------------------------------------------------------
# OAuth 2 token introspection endpoint
# --------------------------------------------------------------------

def _introspected_tokens(obs):
    return _validator_arg_values(obs, 'introspect_token', 0)


def check_introspection_duplicate_conflict(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 200)
    t.assertEqual(obs.parser_view['token'], 'other')
    t.assertEqual(obs.parser_view['duplicate_params'], ['token'])
    t.assertEqual(_introspected_tokens(obs), ['other'])


def check_introspection_empty_hint(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 200)
    t.assertEqual(obs.parser_view['token_type_hint'], '')
    t.assertEqual(_validator_arg_values(obs, 'introspect_token', 1), [''])


def check_introspection_encoded_nul(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 200)
    t.assertEqual(_introspected_tokens(obs), ['föo bar\x00'])


def check_introspection_unicode_nfd(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 200)
    expected = unicodedata.normalize('NFD', 'föo bar')
    t.assertEqual(obs.parser_view['token'], expected)
    t.assertEqual(_introspected_tokens(obs), [expected])


def check_introspection_overlong(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 200)
    t.assertEqual(_introspected_tokens(obs), ['föo bar' + 'x' * 8192])


class IntrospectionEndpointCorpus(MutationCorpusBase, TestCase):
    tracked_params = ('token', 'token_type_hint')
    protected_methods = ('introspect_token',)

    def setUp(self):
        self.endpoint, self.validator = make_introspect_endpoint()
        self.baseline = introspection_baseline()

    def run_endpoint(self, wire):
        return self.endpoint.create_introspect_response(
            wire.uri, http_method=wire.method, body=wire.body,
            headers=wire.headers)

    mutations = (
        Mutation('percent-encoding-case', SAME, corpus.flip_percent_case),
        Mutation('plus-vs-percent20', SAME, corpus.plus_to_percent20),
        Mutation('parameter-reorder', SAME, corpus.reorder),
        Mutation('duplicate-token-last-value-wins', PROTOCOL,
                 corpus.duplicate('token', 'other'),
                 check=check_introspection_duplicate_conflict,
                 note='introspection has no duplicate check'),
        Mutation('token-moved-to-query', REJECT,
                 corpus.move_to_query('token'),
                 note='introspection rejects query parameters on POST'),
        Mutation('empty-token-type-hint', PROTOCOL,
                 corpus.empty_value('token_type_hint'),
                 check=check_introspection_empty_hint),
        Mutation('raw-nul-in-token', REJECT, corpus.raw_nul('token')),
        Mutation('encoded-nul-in-token', PROTOCOL,
                 corpus.encoded_nul('token'),
                 check=check_introspection_encoded_nul),
        Mutation('unicode-nfd-token', PROTOCOL, corpus.unicode_nfd('token'),
                 check=check_introspection_unicode_nfd),
        Mutation('overlong-token', PROTOCOL, corpus.overlong('token'),
                 check=check_introspection_overlong),
    )


# --------------------------------------------------------------------
# OAuth 2 device authorization endpoint (RFC 8628)
# --------------------------------------------------------------------

def check_device_encoded_nul_scope(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 200)
    t.assertEqual(obs.parser_view['scope'], 'bar baz\x00')


def check_device_unicode_nfd_client_id(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 200)
    expected = unicodedata.normalize('NFD', 'föo')
    t.assertEqual(obs.parser_view['client_id'], expected)
    t.assertEqual(_validator_arg_values(obs, 'validate_client_id', 0),
                  [expected])


def check_device_overlong_client_id(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 200)
    t.assertEqual(obs.parser_view['client_id'], 'föo' + 'x' * 8192)


class DeviceAuthorizationEndpointCorpus(MutationCorpusBase, TestCase):
    tracked_params = ('client_id', 'scope')
    protected_methods = ()

    def setUp(self):
        self.endpoint, self.validator = make_device_endpoint()
        self.baseline = device_baseline()

    def run_endpoint(self, wire):
        target = ('oauthlib.oauth2.rfc8628.endpoints.device_authorization'
                  '.generate_token')
        with mock.patch(target, new=lambda: 'DEVICE-CODE'):
            return self.endpoint.create_device_authorization_response(
                wire.uri, http_method=wire.method, body=wire.body,
                headers=wire.headers)

    mutations = (
        Mutation('percent-encoding-case', SAME, corpus.flip_percent_case),
        Mutation('plus-vs-percent20', SAME, corpus.plus_to_percent20),
        Mutation('parameter-reorder', SAME, corpus.reorder),
        Mutation('duplicate-client-id', REJECT,
                 corpus.duplicate('client_id'),
                 note='fatal: client_id is in the enforced duplicate set'),
        Mutation('duplicate-scope', REJECT, corpus.duplicate('scope'),
                 note='fatal: scope is in the enforced duplicate set'),
        Mutation('client-id-moved-to-query', SAME,
                 corpus.move_to_query('client_id'),
                 note='device endpoint accepts parameters from the query'),
        Mutation('content-type-dropped', REJECT,
                 corpus.drop_header('Content-Type'),
                 note='device endpoint enforces the form content type'),
        Mutation('raw-nul-in-client-id', REJECT, corpus.raw_nul('client_id')),
        Mutation('encoded-nul-in-scope', PROTOCOL,
                 corpus.encoded_nul('scope'),
                 check=check_device_encoded_nul_scope),
        Mutation('unicode-nfd-client-id', PROTOCOL,
                 corpus.unicode_nfd('client_id'),
                 check=check_device_unicode_nfd_client_id),
        Mutation('overlong-client-id', PROTOCOL,
                 corpus.overlong('client_id'),
                 check=check_device_overlong_client_id),
    )


# --------------------------------------------------------------------
# OpenID Connect authorization endpoint (code flow, openid scope)
# --------------------------------------------------------------------

def check_oidc_duplicate_nonce(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 302)
    t.assertEqual(obs.parser_view['nonce'], 'second-nonce')
    t.assertEqual(obs.parser_view['duplicate_params'], ['nonce'])


def check_oidc_unicode_nfd_state(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 302)
    t.assertEqual(obs.parser_view['state'],
                  unicodedata.normalize('NFD', 'café'))
    t.assertIn('state=cafe%CC%81', obs.headers['Location'])


def check_oidc_encoded_nul_nonce(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 302)
    t.assertEqual(obs.parser_view['nonce'], 'n-0S6_WzA2Mj\x00')


def check_oidc_overlong_nonce(t, baseline, obs):
    t.assertIsNone(obs.exception)
    t.assertEqual(obs.status, 302)
    t.assertEqual(obs.parser_view['nonce'], 'n-0S6_WzA2Mj' + 'x' * 8192)


class OIDCAuthorizationEndpointCorpus(MutationCorpusBase, TestCase):
    tracked_params = ('response_type', 'client_id', 'redirect_uri',
                      'scope', 'state', 'nonce')
    protected_methods = ('save_authorization_code', 'save_token')

    def setUp(self):
        self.endpoint, self.validator = make_authorization_endpoint(oidc=True)
        self.baseline = oidc_baseline()

    def run_endpoint(self, wire):
        with mock.patch('oauthlib.common.generate_token', new=lambda: 'abc'):
            return self.endpoint.create_authorization_response(
                wire.uri, http_method=wire.method, body=wire.body,
                headers=wire.headers, scopes=['openid', 'profile'])

    mutations = (
        Mutation('percent-encoding-case', SAME, corpus.flip_percent_case),
        Mutation('plus-vs-percent20', SAME, corpus.plus_to_percent20),
        Mutation('parameter-reorder', SAME, corpus.reorder),
        Mutation('nonce-moved-to-body', SAME, move_nonce_to_post_body,
                 note='OIDC authorization endpoint merges query and body'),
        Mutation('duplicate-client-id', REJECT,
                 corpus.duplicate('client_id'),
                 note='fatal: rejected before any redirect'),
        Mutation('duplicate-nonce-last-value-wins', PROTOCOL,
                 corpus.duplicate('nonce', 'second-nonce'),
                 check=check_oidc_duplicate_nonce,
                 note='nonce is not in the enforced duplicate set'),
        Mutation('raw-nul-in-client-id', REJECT, corpus.raw_nul('client_id')),
        Mutation('encoded-nul-in-nonce', PROTOCOL,
                 corpus.encoded_nul('nonce'),
                 check=check_oidc_encoded_nul_nonce),
        Mutation('unicode-nfd-state', PROTOCOL, corpus.unicode_nfd('state'),
                 check=check_oidc_unicode_nfd_state),
        Mutation('overlong-nonce', PROTOCOL, corpus.overlong('nonce'),
                 check=check_oidc_overlong_nonce),
        Mutation('error-redirect', PROTOCOL,
                 corpus.set_param('response_type', 'bogus'),
                 check=check_error_redirect,
                 note='OIDC shares the OAuth2 error redirect rule'),
    )


# --------------------------------------------------------------------
# OAuth 1 signature-only endpoint (RFC 5849)
# --------------------------------------------------------------------

class OAuth1SignatureCorpus(MutationCorpusBase, TestCase):
    tracked_params = ('foo', 'a', 'b')
    protected_methods = ()

    def setUp(self):
        self.endpoint, self.validator = make_oauth1_endpoint()
        self.baseline = oauth1_baseline()

    def run_endpoint(self, wire):
        valid, _request = self.endpoint.validate_request(
            wire.uri, http_method=wire.method, body=wire.body,
            headers=wire.headers)
        return {}, valid, 200 if valid else 401

    mutations = (
        Mutation('parameter-reorder', SAME, corpus.reorder,
                 note='OAuth1 sorts parameters into the signature base '
                      'string'),
        Mutation('percent-encoding-case', SAME, corpus.flip_percent_case,
                 note='signature normalization re-encodes percent bytes'),
        Mutation('plus-vs-percent20', SAME, corpus.plus_to_percent20),
        Mutation('oauth-params-header-to-query', SAME,
                 corpus.oauth1_header_to_query,
                 note='OAuth1 accepts protocol parameters from a single '
                      'alternative source'),
        Mutation('duplicate-parameter', REJECT, corpus.duplicate('foo'),
                 note='duplicates change the signature base string'),
        Mutation('duplicate-oauth-parameter', REJECT,
                 corpus.duplicate('oauth_consumer_key', 'foo'),
                 note='oauth_ parameters must come from a single source'),
        Mutation('raw-nul-in-parameter', REJECT, corpus.raw_nul('foo')),
        Mutation('encoded-nul-in-parameter', REJECT,
                 corpus.encoded_nul('foo'),
                 note='any value change invalidates the signature'),
        Mutation('unicode-nfd-parameter', REJECT, corpus.unicode_nfd('foo'),
                 note='NFC and NFD are distinct under the signature'),
        Mutation('overlong-parameter', REJECT, corpus.overlong('foo')),
    )


# --------------------------------------------------------------------
# Degenerate oracles: the corpus must distinguish them
# --------------------------------------------------------------------

class DegenerateOracleTest(TestCase):

    def test_last_value_wins_matches_first_value_wins_does_not(self):
        endpoint, validator = make_token_endpoint()
        wire = corpus.duplicate('code', 'xyz')(token_baseline())
        obs = corpus.observe(
            lambda w: endpoint.create_token_response(
                w.uri, http_method=w.method, body=w.body, headers=w.headers),
            wire, validator, ('code',))
        pairs = wire.body_pairs
        self.assertNotEqual(corpus.first_value_wins(pairs),
                            corpus.last_value_wins(pairs))
        self.assertEqual(obs.parser_view['code'],
                         corpus.last_value_wins(pairs)['code'])
        self.assertNotEqual(obs.parser_view['code'],
                            corpus.first_value_wins(pairs)['code'])
        self.assertEqual(
            [args[1] for name, args, _ in obs.validator_calls
             if name == 'validate_code'],
            [corpus.last_value_wins(pairs)['code']])

    def test_error_redirect_distinguishes_fatal_from_normal(self):
        endpoint, validator = make_authorization_endpoint()
        baseline = authorization_baseline()

        def run(wire):
            with mock.patch('oauthlib.common.generate_token',
                            new=lambda: 'abc'):
                return endpoint.create_authorization_response(
                    wire.uri, http_method=wire.method, body=wire.body,
                    headers=wire.headers, scopes=['all', 'of', 'them'])

        normal = corpus.set_param('response_type', 'bogus')(baseline)
        obs = corpus.observe(run, normal, validator, ('response_type',))
        location = obs.headers.get('Location', '')
        self.assertTrue(location.startswith('https://back.to/me?'))
        self.assertIn('error=unsupported_response_type', location)
        self.assertNotIn('save_authorization_code',
                         [c[0] for c in obs.validator_calls])

        fatal = corpus.duplicate('client_id')(baseline)
        obs = corpus.observe(run, fatal, validator, ('client_id',))
        self.assertEqual(obs.exception, 'InvalidRequestFatalError')
        self.assertNotIn('Location', obs.headers)
        self.assertNotIn('save_authorization_code',
                         [c[0] for c in obs.validator_calls])


class CorpusDeterminismTest(TestCase):

    def test_reorder_is_seeded_and_effective(self):
        wire = token_baseline()
        self.assertEqual(corpus.reorder(wire), corpus.reorder(wire))
        self.assertNotEqual(corpus.reorder(wire), wire)

    def test_relations_are_closed(self):
        for cls in (TokenEndpointCorpus, AuthorizationEndpointCorpus,
                    RevocationEndpointCorpus, IntrospectionEndpointCorpus,
                    DeviceAuthorizationEndpointCorpus,
                    OIDCAuthorizationEndpointCorpus, OAuth1SignatureCorpus):
            for mutation in cls.mutations:
                self.assertIn(mutation.relation, corpus.RELATIONS)
