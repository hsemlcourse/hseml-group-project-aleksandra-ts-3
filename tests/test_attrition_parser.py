import pytest

from src.attrition_parser import parse_ibm_attrition_csv


def test_parse_ibm_attrition_csv_minimal():
    csv_text = (
        'Age,Attrition,Department\n'
        '30,Yes,Sales\n'
        '45,No,R&D\n'
    )
    df = parse_ibm_attrition_csv(csv_text)
    assert len(df) == 2
    assert list(df.columns) == ['Age', 'Attrition', 'Department']
    assert df.iloc[0]['Attrition'] == 'Yes'
    assert df.iloc[1]['Age'] == '45'


def test_parse_ibm_attrition_csv_empty_raises():
    with pytest.raises(ValueError, match='пустой'):
        parse_ibm_attrition_csv('')
