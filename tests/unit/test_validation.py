from src.validation.validation import CodeValidator

def test_valid_syntax():
    res = CodeValidator.validate("def foo():\n    return 1")
    assert res.is_valid

def test_invalid_syntax():
    res = CodeValidator.validate("def foo() return 1")
    assert not res.is_valid
