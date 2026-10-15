"""Streamlit interface for term-deposit subscription predictions."""

from __future__ import annotations

import os

import streamlit as st

from src.demo.client import DemoApiError, build_prediction_payload, request_prediction

JOBS = [
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
]
MARITAL_STATUSES = ["divorced", "married", "single"]
EDUCATION_LEVELS = ["primary", "secondary", "tertiary", "unknown"]
YES_NO = ["no", "yes"]
CONTACT_TYPES = ["unknown", "telephone", "cellular"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
PREVIOUS_OUTCOMES = ["failure", "other", "success", "unknown"]

RUSSIAN_OPTIONS = {
    "admin.": "административная работа",
    "blue-collar": "рабочая профессия",
    "entrepreneur": "предприниматель",
    "housemaid": "домашнее хозяйство",
    "management": "руководитель",
    "retired": "пенсионер",
    "self-employed": "самозанятый",
    "services": "сфера услуг",
    "student": "студент",
    "technician": "технический специалист",
    "unemployed": "безработный",
    "divorced": "разведён(а)",
    "married": "в браке",
    "single": "не в браке",
    "primary": "начальное",
    "secondary": "среднее",
    "tertiary": "высшее",
    "no": "нет",
    "yes": "да",
    "telephone": "стационарный телефон",
    "cellular": "мобильный телефон",
    "failure": "неуспешный",
    "other": "другой",
    "success": "успешный",
    "unknown": "неизвестно",
}


def option_label(value: str) -> str:
    """Return a Russian label while preserving the API value."""

    return RUSSIAN_OPTIONS.get(value, value)


def main() -> None:
    """Render the prediction form and its result."""

    st.set_page_config(page_title="Прогноз отклика клиента", page_icon="🏦", layout="centered")
    st.title("Прогноз отклика на банковское предложение")
    st.write(
        "Заполните сведения о клиенте и маркетинговом контакте. Модель оценит вероятность оформления срочного депозита."
    )

    api_url = os.getenv("API_URL", "http://127.0.0.1:8000")
    st.caption(f"Inference API: {api_url}")

    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            st.subheader("Клиент")
            age = st.number_input("Возраст", min_value=0, max_value=120, value=40, step=1)
            job = st.selectbox("Профессия", JOBS, format_func=option_label)
            marital = st.selectbox("Семейное положение", MARITAL_STATUSES, format_func=option_label)
            education = st.selectbox("Образование", EDUCATION_LEVELS, format_func=option_label)
            default = st.selectbox(
                "Есть просрочка по кредиту",
                YES_NO,
                format_func=option_label,
                help="Признак наличия кредитной просрочки.",
            )
            balance = st.number_input(
                "Среднегодовой баланс, €",
                value=0,
                step=100,
                help="Баланс может быть отрицательным.",
            )
            housing = st.selectbox("Есть жилищный кредит", YES_NO, format_func=option_label)
            loan = st.selectbox("Есть потребительский кредит", YES_NO, format_func=option_label)

        with right:
            st.subheader("Маркетинговый контакт")
            contact = st.selectbox("Способ связи", CONTACT_TYPES, format_func=option_label)
            day = st.number_input("День месяца", min_value=1, max_value=31, value=15, step=1)
            month = st.selectbox("Месяц", MONTHS)
            duration = st.number_input(
                "Продолжительность контакта, секунд",
                min_value=0,
                value=180,
                step=1,
            )
            campaign = st.number_input(
                "Контактов в текущей кампании",
                min_value=0,
                value=1,
                step=1,
            )
            pdays = st.number_input(
                "Дней после предыдущего контакта",
                value=-1,
                step=1,
                help="Значение −1 означает, что ранее с клиентом не связывались.",
            )
            previous = st.number_input(
                "Контактов до текущей кампании",
                min_value=0,
                value=0,
                step=1,
            )
            poutcome = st.selectbox(
                "Результат предыдущей кампании",
                PREVIOUS_OUTCOMES,
                format_func=option_label,
            )

        submitted = st.form_submit_button("Получить прогноз", type="primary", use_container_width=True)

    if not submitted:
        return

    payload = build_prediction_payload(
        age=int(age),
        job=job,
        marital=marital,
        education=education,
        default=default,
        balance=int(balance),
        housing=housing,
        loan=loan,
        contact=contact,
        day=int(day),
        month=month,
        duration=int(duration),
        campaign=int(campaign),
        pdays=int(pdays),
        previous=int(previous),
        poutcome=poutcome,
    )

    try:
        with st.spinner("Модель рассчитывает вероятность..."):
            result = request_prediction(payload, api_url)
    except DemoApiError as exc:
        st.error(str(exc))
        return

    st.subheader("Результат")
    st.metric("Вероятность оформления депозита", f"{result['subscription_probability']:.2%}")
    if result["prediction"] == "yes":
        st.success("Клиент, вероятно, откликнется на предложение.")
    else:
        st.info("Клиент, вероятно, не откликнется на предложение.")
    st.caption(f"Версия модели: {result['model_version']}")


if __name__ == "__main__":
    main()
