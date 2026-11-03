from rag import chunk_text


def test_empty():
    assert chunk_text(' ') == []


def test_overlapping():
    chunks = chunk_text('abcdefghijklmnopqrstuvwxyz', chunk_size=10, overlap=3)
    assert chunks[0] == 'abcdefghij'
    assert chunks[1][:3] == chunks[0][-3:]
    assert chunks[-1].endswith('xyz')
