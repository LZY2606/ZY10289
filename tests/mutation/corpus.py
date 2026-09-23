# -*- coding: utf-8 -*-
"""Protocol-aware mutation corpus for oauthlib endpoint parsing.

Mutations are derived from a known-good ("baseline") wire request and
cover percent-encoding case, plus/space equivalence, parameter
reordering, duplication, query/body relocation, empty values, NUL
bytes, Unicode normalization forms and overlong values.

For every mutated request the corpus records, without performing real
HTTP:

* the parser view (what ``oauthlib.common.Request`` extracts),
* the validator calls the endpoint performs,
* the wire response (status, headers, body) or the raised exception.

Each corpus entry declares its expected relation to the baseline:

* ``SAME`` -- the mutation is a pure re-encoding and must be accepted
  with an identical parser view, identical validator calls and an
  equivalent response.
* ``REJECT`` -- the mutation must fail; token-issuing validator
  methods (e.g. ``save_token``) must not be reached.
* ``PROTOCOL`` -- protocol-specific behavior pinned by an explicit
  check callable (e.g. OAuth1 signature rules, duplicate handling,
  verbatim redirect_uri comparison).

Three intentionally degenerate oracles (first-value-wins,
last-value-wins and error-redirect) are provided so the corpus can
prove it distinguishes between them.

All randomness flows through a fixed seed so the corpus is
deterministic and reducible.
"""
import json
import random
import re
import unicodedata
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlparse, urlunparse

from oauthlib.common import Request, unquote, urldecode, urlencode

#: Fixed seed for every randomized mutation; keeps the corpus
#: deterministic and allows shrinking to a reproducible subset.
SEED = 1293

#: Expected relations between a mutated request and its baseline.
SAME = 'same'
REJECT = 'reject'
PROTOCOL = 'protocol-specific'
RELATIONS = (SAME, REJECT, PROTOCOL)

_HEX_BYTE = re.compile(r'%[0-9A-Fa-f]{2}')


@dataclass
class WireRequest:
    """A serializable HTTP request as it arrives at an endpoint."""
    method: str
    uri: str
    body: str = None
    headers: dict = field(default_factory=dict)

    @property
    def query_pairs(self):
        return parse_qsl(urlparse(self.uri).query, keep_blank_values=True)

    @property
    def body_pairs(self):
        return urldecode(self.body) if self.body else []

    def with_query(self, pairs):
        parts = urlparse(self.uri)
        uri = urlunparse(parts._replace(query=urlencode(pairs)))
        return WireRequest(self.method, uri, self.body, dict(self.headers))

    def with_body(self, pairs):
        return WireRequest(self.method, self.uri, urlencode(pairs),
                           dict(self.headers))

    def with_headers(self, headers):
        return WireRequest(self.method, self.uri, self.body, headers)


def _pairs_of(wire):
    """Return (pairs, setter) for the parameter source of a request."""
    if wire.body:
        return wire.body_pairs, wire.with_body
    return wire.query_pairs, wire.with_query


def _first_value(pairs, param):
    for key, value in pairs:
        if key == param:
            return value
    raise KeyError(param)


# --------------------------------------------------------------------
# Mutation operators
# --------------------------------------------------------------------

def flip_percent_case(wire):
    """Flip the hex case of the first percent-encoded byte (%3A <-> %3a)."""
    def flip_first(text):
        for match in _HEX_BYTE.finditer(text):
            flipped = match.group(0).swapcase()
            if flipped != match.group(0):  # skip digit-only bytes like %20
                return text[:match.start()] + flipped + text[match.end():]
        return text

    if wire.body and _HEX_BYTE.search(wire.body):
        return WireRequest(wire.method, wire.uri, flip_first(wire.body),
                           dict(wire.headers))
    return WireRequest(wire.method, flip_first(wire.uri), wire.body,
                       dict(wire.headers))


def plus_to_percent20(wire):
    """Encode a space as %20 instead of + (or vice versa)."""
    def swap(text):
        if '+' in text:
            return text.replace('+', '%20')
        return text.replace('%20', '+', 1)

    if wire.body and ('+' in wire.body or '%20' in wire.body):
        return WireRequest(wire.method, wire.uri, swap(wire.body),
                           dict(wire.headers))
    return WireRequest(wire.method, swap(wire.uri), wire.body,
                       dict(wire.headers))


def reorder(wire, seed=SEED):
    """Shuffle parameter order with a fixed seed."""
    rng = random.Random(seed)  # noqa: S311 - deterministic test corpus
    pairs, setter = _pairs_of(wire)
    pairs = list(pairs)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    if shuffled == pairs and len(pairs) > 1:
        shuffled = pairs[::-1]
    return setter(shuffled)


def duplicate(param, value=None):
    """Append a second occurrence of param (same or conflicting value)."""
    def apply(wire):
        pairs, setter = _pairs_of(wire)
        val = _first_value(pairs, param) if value is None else value
        return setter(list(pairs) + [(param, val)])
    return apply


def move_to_query(param):
    """Relocate a parameter from the body to the query string."""
    def apply(wire):
        pairs = list(wire.body_pairs)
        val = _first_value(pairs, param)
        pairs = [(k, v) for k, v in pairs if k != param]
        moved = wire.with_body(pairs)
        return moved.with_query(moved.query_pairs + [(param, val)])
    return apply


def move_to_body(param):
    """Relocate a parameter from the query string to the body."""
    def apply(wire):
        pairs = list(wire.query_pairs)
        val = _first_value(pairs, param)
        moved = wire.with_query([(k, v) for k, v in pairs if k != param])
        return moved.with_body(list(moved.body_pairs) + [(param, val)])
    return apply


def set_param(param, value):
    """Overwrite the first occurrence of param with a new value."""
    def apply(wire):
        pairs, setter = _pairs_of(wire)
        seen = [False]

        def replace(pair):
            key, _val = pair
            if key == param and not seen[0]:
                seen[0] = True
                return (key, value)
            return pair
        return setter([replace(p) for p in pairs])
    return apply


def empty_value(param):
    """Send param with an empty value."""
    return set_param(param, '')


def raw_nul(param):
    """Inject a literal (unencoded) NUL byte into the serialized wire."""
    def apply(wire):
        if wire.body and param + '=' in wire.body:
            body = wire.body.replace(param + '=', param + '=\x00', 1)
            return WireRequest(wire.method, wire.uri, body,
                               dict(wire.headers))
        uri = wire.uri.replace(param + '=', param + '=\x00', 1)
        return WireRequest(wire.method, uri, wire.body, dict(wire.headers))
    return apply


def encoded_nul(param):
    """Append a percent-encoded NUL (%00) to a parameter value."""
    def apply(wire):
        pairs, _setter = _pairs_of(wire)
        val = _first_value(pairs, param)
        return set_param(param, val + '\x00')(wire)
    return apply


def unicode_nfd(param):
    """Replace the value of param with its NFD normalization form."""
    def apply(wire):
        pairs, _setter = _pairs_of(wire)
        val = _first_value(pairs, param)
        nfd = unicodedata.normalize('NFD', val)
        assert nfd != val, 'baseline value has no distinct NFD form'
        return set_param(param, nfd)(wire)
    return apply


def overlong(param, size=8192):
    """Inflate a parameter value with trailing padding."""
    def apply(wire):
        pairs, _setter = _pairs_of(wire)
        val = _first_value(pairs, param)
        return set_param(param, val + 'x' * size)(wire)
    return apply


def drop_header(name):
    """Remove a header from the request."""
    def apply(wire):
        headers = {k: v for k, v in wire.headers.items()
                   if k.lower() != name.lower()}
        return wire.with_headers(headers)
    return apply


def oauth1_header_to_query(wire):
    """Move OAuth1 protocol parameters from the Authorization header
    into the query string (multi-source relocation)."""
    auth = wire.headers.get('Authorization', '')
    assert auth.startswith('OAuth '), 'expected OAuth header'
    pairs = []
    for item in auth[len('OAuth '):].split(','):
        key, _, val = item.strip().partition('=')
        pairs.append((key, unquote(val.strip('"'))))
    headers = {k: v for k, v in wire.headers.items()
               if k.lower() != 'authorization'}
    moved = WireRequest(wire.method, wire.uri, wire.body, headers)
    return moved.with_query(moved.query_pairs + pairs)


# --------------------------------------------------------------------
# Corpus entries
# --------------------------------------------------------------------

@dataclass
class Mutation:
    """One corpus entry: a named mutation with its expected relation."""
    name: str
    relation: str
    apply: object
    note: str = ''
    check: object = None  # PROTOCOL: check(testcase, baseline, mutated)

    def __post_init__(self):
        assert self.relation in RELATIONS, self.relation
        if self.relation == PROTOCOL:
            assert callable(self.check), self.name


# --------------------------------------------------------------------
# Observation harness
# --------------------------------------------------------------------

def parser_view(wire, params):
    """Record what oauthlib.common.Request extracts from the wire."""
    try:
        req = Request(wire.uri, http_method=wire.method,
                      body=wire.body, headers=dict(wire.headers))
    except Exception as exc:  # noqa: BLE001 - record any parse failure
        return {'parse_error': '%s: %s' % (type(exc).__name__, exc)}
    view = {p: getattr(req, p, None) for p in params}
    try:
        view['duplicate_params'] = sorted(req.duplicate_params)
    except ValueError:
        view['duplicate_params'] = '<unparseable>'
    return view


def _sanitize(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    return '<%s>' % type(value).__name__


def validator_calls(validator):
    """Record the validator method calls made while serving a request."""
    calls = []
    for invoked in validator.mock_calls:
        name, args, kwargs = invoked
        if not name or '.' in name or name.startswith('_'):
            continue
        calls.append((name,
                      tuple(_sanitize(a) for a in args),
                      tuple(sorted((k, _sanitize(v))
                                   for k, v in kwargs.items()))))
    return calls


@dataclass
class Observation:
    """Recorded outcome of serving one (possibly mutated) request."""
    parser_view: dict
    validator_calls: list
    status: object
    headers: dict
    body: object
    exception: str = None


def observe(run, wire, validator, params):
    """Serve wire through run() and record the full observation."""
    validator.reset_mock()
    view = parser_view(wire, params)
    headers, body, status, exception = {}, None, None, None
    try:
        headers, body, status = run(wire)
    except Exception as exc:  # noqa: BLE001 - record any endpoint failure
        exception = type(exc).__name__
    return Observation(view, validator_calls(validator), status,
                       dict(headers or {}), body, exception)


# --------------------------------------------------------------------
# Relation assertions
# --------------------------------------------------------------------

def normalized_body(body):
    if isinstance(body, str):
        try:
            return ('json', json.loads(body))
        except ValueError:
            return ('raw', body)
    return ('obj', body)


def assert_same(testcase, baseline, mutated):
    """The mutated request must be indistinguishable from the baseline."""
    testcase.assertIsNone(baseline.exception)
    testcase.assertIsNone(mutated.exception)
    testcase.assertEqual(mutated.status, baseline.status)
    testcase.assertEqual(mutated.parser_view, baseline.parser_view)
    testcase.assertEqual(mutated.validator_calls, baseline.validator_calls)
    testcase.assertEqual(mutated.headers, baseline.headers)
    testcase.assertEqual(normalized_body(mutated.body),
                         normalized_body(baseline.body))


def response_is_error(obs):
    if obs.exception is not None:
        return True
    if isinstance(obs.status, int) and obs.status >= 400:
        return True
    if obs.status is False:  # OAuth1 endpoints signal failure with False
        return True
    body = normalized_body(obs.body)
    if body[0] == 'json' and isinstance(body[1], dict) and 'error' in body[1]:
        return True
    location = obs.headers.get('Location', '')
    return 'error=' in location


def assert_reject(testcase, obs, protected_methods=()):
    """The mutated request must fail before any token-issuing call."""
    testcase.assertTrue(
        response_is_error(obs),
        'expected rejection, got status=%r body=%r' % (obs.status, obs.body))
    called = [name for name, _args, _kwargs in obs.validator_calls]
    for method in protected_methods:
        testcase.assertNotIn(
            method, called,
            'rejected request reached token-issuing validator %s' % method)


# --------------------------------------------------------------------
# Intentionally degenerate oracles
# --------------------------------------------------------------------

def first_value_wins(pairs):
    """Degenerate oracle: the first occurrence of a parameter wins."""
    view = {}
    for key, value in pairs:
        if key not in view:
            view[key] = value
    return view


def last_value_wins(pairs):
    """Degenerate oracle: the last occurrence of a parameter wins."""
    return dict(pairs)


def error_redirect(redirect_uri, error, state=None):
    """Degenerate oracle: errors are reported by redirecting to the
    registered redirect_uri with error parameters appended."""
    location = redirect_uri + '?error=' + error
    if state is not None:
        location += '&state=' + state
    return location
