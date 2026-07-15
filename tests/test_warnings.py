import warnings

from credit_risk import silence_library_warnings


def test_silence_library_warnings_mutes_user_warnings() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        silence_library_warnings()
        warnings.warn("feature name noise", UserWarning, stacklevel=1)

    assert not [record for record in caught if issubclass(record.category, UserWarning)]


def test_silence_library_warnings_leaves_other_warnings_visible() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        silence_library_warnings()
        warnings.warn("real problem", FutureWarning, stacklevel=1)

    assert [record for record in caught if issubclass(record.category, FutureWarning)]
