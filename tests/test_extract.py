from autoscale_rl.extract import parse_line

REAL = ('199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] '
        '"GET /history/apollo/ HTTP/1.0" 200 6246')
REAL_POST = ('alice.example.com - - [05/Jun/1999:10:01:00 -0400] '
             '"POST /cgi-bin/order HTTP/1.0" 200 111')
MALFORMED = 'not a log line at all'
NO_QUOTE = ('10.0.0.1 - - [01/Jul/1995:00:00:02 -0400] GET /x HTTP/1.0')


def test_parses_common_format():
    assert parse_line(REAL) == (1995, 7, 1, 0, 0)


def test_parses_non_get_method():
    assert parse_line(REAL_POST) == (1999, 6, 5, 10, 1)


def test_skips_malformed():
    assert parse_line(MALFORMED) is None
    assert parse_line(NO_QUOTE) is None
