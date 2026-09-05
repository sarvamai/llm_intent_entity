from llm_intent_entity.utilities import lang_to_code, indic_langs


def test_assamese_maps_to_as_not_odia():
    # 'or' is Odia's ISO 639-1 code; Assamese is 'as'
    assert lang_to_code["assamese"] == "as"


def test_assamese_reaches_indic_normalizer():
    # 'as' must be in indic_langs or Assamese skips the Indic normalizer
    assert "as" in indic_langs
