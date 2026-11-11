import pytest

from burmeseai.train import mean_init_resize, mean_init_row, merge_rows


def test_per_piece_mean_on_a_tiny_tied_embedding():
    embedding = [
        [1.0, 1.0],
        [3.0, 5.0],
        [0.0, 2.0],
    ]
    grown, output = mean_init_resize(
        embedding,
        [[0, 1], [2]],
        embedding,
        tie_word_embeddings=True,
    )
    assert output is grown
    assert mean_init_row(embedding, [0, 1]) == [2.0, 3.0]
    assert grown[3] == [2.0, 3.0]
    assert grown[4] == [0.0, 2.0]
    assert grown[:3] == embedding
    assert len(grown) == 5


def test_untied_mean_uses_each_matrix():
    incoming = [
        [1.0, 1.0],
        [3.0, 5.0],
    ]
    outgoing = [
        [10.0, 0.0],
        [0.0, 10.0],
    ]
    grown_in, grown_out = mean_init_resize(
        incoming,
        [[0, 1]],
        outgoing,
        tie_word_embeddings=False,
    )
    assert grown_in is not grown_out
    assert grown_in[2] == [2.0, 3.0]
    assert grown_out[2] == [5.0, 5.0]
    assert grown_in[:2] == incoming
    assert grown_out[:2] == outgoing


def test_tied_flag_rejects_two_matrices():
    incoming = [[1.0, 0.0], [0.0, 1.0]]
    outgoing = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError, match="one matrix"):
        mean_init_resize(incoming, [[0]], outgoing, tie_word_embeddings=True)


def test_empty_piece_list_is_rejected():
    embedding = [[1.0, 2.0]]
    with pytest.raises(ValueError, match="source piece"):
        mean_init_resize(embedding, [[]], embedding, tie_word_embeddings=True)


def test_merge_is_half_and_half_by_default():
    mixed = merge_rows([[0.0, 2.0]], [[4.0, 6.0]])
    assert mixed == [[2.0, 4.0]]
