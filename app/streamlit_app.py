import sys
from copy import deepcopy
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.theme import LAVENDER, NAVY, RISK_BG, RISK_COLORS, TEAL, THEME_CSS, WHITE
from src.inference import AttritionPredictor, load_default_employee

st.set_page_config(
    page_title='HR Attrition: прогноз оттока',
    page_icon='👩‍💻',
    layout='wide',
    initial_sidebar_state='collapsed',
)

st.markdown(THEME_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return AttritionPredictor()


def high_risk_profile(base):
    p = deepcopy(base)
    p.update(
        {
            'OverTime': 'Yes',
            'JobSatisfaction': 1,
            'EnvironmentSatisfaction': 1,
            'WorkLifeBalance': 1,
            'YearsAtCompany': 1.0,
            'YearsInCurrentRole': 0.0,
            'StockOptionLevel': 0,
            'NumCompaniesWorked': 6.0,
        }
    )
    return p


def interpret_risk(level):
    texts = {
        'низкий': 'Профиль близок к удерживаемым сотрудникам. Рекомендуется стандартный мониторинг',
        'средний': 'Есть факторы риска. Имеет смысл обсудить нагрузку, вовлечённость и условия',
        'высокий': 'Высокая вероятность оттока. Приоритет для HR: удержание, 1-on-1, пересмотр условий',
    }
    return texts.get(level, '')


def render_result_card(result):
    prob = result['attrition_probability']
    level = result['risk_level']
    color = RISK_COLORS.get(level, NAVY)
    bg = RISK_BG.get(level, LAVENDER)
    pct = int(round(prob * 100))

    st.markdown(
        f"""
        <div class="result-card risk-{level}">
            <h3>Прогноз оттока</h3>
            <div class="gauge" style="--p:{pct}; --c:{color};">
                <div class="gauge-inner">
                    <span class="gauge-value">{prob:.1%}</span>
                    <span class="gauge-label">вероятность</span>
                </div>
            </div>
            <span class="risk-pill" style="background:{bg};color:{color};">
                {level.upper()} риск
            </span>
            <div class="result-meta">
                <b>Увольнение (50%):</b>
                {'Да: стоит подключить HR' if result['attrition_predicted'] else 'Нет: по текущему порогу'}<br>
                {interpret_risk(level)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    f"""
    <div class="hero">
        <h1 style="color:{WHITE};">Прогноз оттока сотрудников</h1>
        <p style="color:{WHITE};">Введите данные сотрудника и получите оценку риска увольнения</p>
    </div>
    """,
    unsafe_allow_html=True,
)

defaults = load_default_employee()
if not defaults:
    st.stop()

try:
    predictor = load_predictor()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

if 'profile' not in st.session_state:
    st.session_state.profile = deepcopy(defaults)
if 'last_result' not in st.session_state:
    st.session_state.last_result = predictor.explain_one(st.session_state.profile)

p = st.session_state.profile
col_form, col_res = st.columns([1.35, 1], gap='large')

with col_form:
    st.markdown('##### Профиль сотрудника')
    st.caption('Можно заполнить форму примером:')
    ex1, ex2 = st.columns(2, gap='medium')
    with ex1:
        if st.button('Типичный сотрудник', use_container_width=True, type='primary'):
            st.session_state.profile = deepcopy(defaults)
            st.session_state.last_result = predictor.explain_one(st.session_state.profile)
            st.rerun()
    with ex2:
        if st.button('Высокий риск', use_container_width=True, type='primary'):
            st.session_state.profile = high_risk_profile(defaults)
            st.session_state.last_result = predictor.explain_one(st.session_state.profile)
            st.rerun()

    p = st.session_state.profile

    with st.form('predict_form', border=False):
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            p['Age'] = st.number_input('Возраст', 18, 70, int(p['Age']))
            p['Gender'] = st.selectbox('Пол', ['Male', 'Female'], index=['Male', 'Female'].index(p['Gender']))
            p['MaritalStatus'] = st.selectbox(
                'Семейное положение',
                ['Single', 'Married', 'Divorced'],
                index=['Single', 'Married', 'Divorced'].index(p['MaritalStatus']),
            )
        with r1c2:
            p['Department'] = st.selectbox(
                'Отдел',
                ['Research & Development', 'Sales', 'Human Resources'],
                index=['Research & Development', 'Sales', 'Human Resources'].index(p['Department']),
            )
            roles = [
                'Research Director',
                'Sales Representative',
                'Research Scientist',
                'Manufacturing Director',
                'Healthcare Representative',
                'Sales Executive',
                'Manager',
                'Human Resources',
                'Laboratory Technician',
            ]
            p['JobRole'] = st.selectbox('Должность', roles, index=roles.index(p['JobRole']))
            p['OverTime'] = st.selectbox('Переработки', ['Yes', 'No'], index=['Yes', 'No'].index(p['OverTime']))
        with r1c3:
            p['MonthlyIncome'] = st.number_input('Месячный доход', 1000.0, 20000.0, float(p['MonthlyIncome']))
            p['JobSatisfaction'] = st.slider('Удовл. работой', 1, 4, int(p['JobSatisfaction']))
            p['EnvironmentSatisfaction'] = st.slider('Удовл. средой', 1, 4, int(p['EnvironmentSatisfaction']))
            p['WorkLifeBalance'] = st.slider('Work-life balance', 1, 4, int(p['WorkLifeBalance']))
            p['YearsAtCompany'] = st.slider('Стаж в компании', 0.0, 40.0, float(p['YearsAtCompany']))

        with st.expander('Дополнительные поля', expanded=False):
            g1, g2, g3 = st.columns(3)
            with g1:
                p['DistanceFromHome'] = st.number_input('DistanceFromHome', 1, 30, int(p['DistanceFromHome']))
                p['Education'] = st.selectbox('Education', [1, 2, 3, 4, 5], index=int(p['Education']) - 1)
                edu_fields = [
                    'Life Sciences',
                    'Medical',
                    'Marketing',
                    'Technical Degree',
                    'Human Resources',
                    'Other',
                ]
                p['EducationField'] = st.selectbox('EducationField', edu_fields, index=edu_fields.index(p['EducationField']))
                p['BusinessTravel'] = st.selectbox(
                    'BusinessTravel',
                    ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
                    index=['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'].index(p['BusinessTravel']),
                )
            with g2:
                p['DailyRate'] = st.number_input('DailyRate', 100.0, 2000.0, float(p['DailyRate']))
                p['HourlyRate'] = st.number_input('HourlyRate', 30.0, 150.0, float(p['HourlyRate']))
                p['MonthlyRate'] = st.number_input('MonthlyRate', 2000.0, 28000.0, float(p['MonthlyRate']))
                p['JobLevel'] = st.selectbox('JobLevel', [1, 2, 3, 4, 5], index=int(p['JobLevel']) - 1)
                p['PercentSalaryHike'] = st.number_input('PercentSalaryHike', 0.0, 25.0, float(p['PercentSalaryHike']))
            with g3:
                p['JobInvolvement'] = st.selectbox('JobInvolvement', [1, 2, 3, 4], index=int(p['JobInvolvement']) - 1)
                p['PerformanceRating'] = st.selectbox('PerformanceRating', [1, 2, 3, 4], index=int(p['PerformanceRating']) - 1)
                p['RelationshipSatisfaction'] = st.selectbox(
                    'RelationshipSatisfaction', [1, 2, 3, 4], index=int(p['RelationshipSatisfaction']) - 1
                )
                p['StockOptionLevel'] = st.selectbox('StockOptionLevel', [0, 1, 2, 3], index=int(p['StockOptionLevel']))
                p['TotalWorkingYears'] = st.number_input('TotalWorkingYears', 0.0, 40.0, float(p['TotalWorkingYears']))
                p['YearsInCurrentRole'] = st.number_input('YearsInCurrentRole', 0.0, 20.0, float(p['YearsInCurrentRole']))
                p['YearsSinceLastPromotion'] = st.number_input(
                    'YearsSinceLastPromotion', 0.0, 20.0, float(p['YearsSinceLastPromotion'])
                )
                p['YearsWithCurrManager'] = st.number_input(
                    'YearsWithCurrManager', 0.0, 20.0, float(p['YearsWithCurrManager'])
                )
                p['NumCompaniesWorked'] = st.number_input('NumCompaniesWorked', 0.0, 10.0, float(p['NumCompaniesWorked']))
                p['TrainingTimesLastYear'] = st.number_input(
                    'TrainingTimesLastYear', 0.0, 6.0, float(p['TrainingTimesLastYear'])
                )

        submitted = st.form_submit_button('Рассчитать риск оттока', type='primary', use_container_width=True)

    st.session_state.profile = p
    if submitted:
        st.session_state.last_result = predictor.explain_one(p)

with col_res:
    if st.session_state.last_result:
        render_result_card(st.session_state.last_result)
    else:
        st.markdown(
            '<div class="panel-soft">Заполните форму и нажмите "Рассчитать риск оттока".</div>',
            unsafe_allow_html=True,
        )
