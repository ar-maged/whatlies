import pytest

from whatlies.language import BytePairLang


@pytest.fixture()
def lang():
    return BytePairLang("en", vs=1000, dim=100)


def test_single_token_words(lang):
    assert lang["red"].vector.shape == (100, )
    assert len(lang[["red", "blue"]]) == 2


@pytest.mark.parametrize("item", [2, .12341])
def test_raise_error(lang, item):
    with pytest.raises(ValueError):
        _ = lang[item]
