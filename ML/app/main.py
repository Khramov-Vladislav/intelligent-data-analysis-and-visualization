"""Taipy-приложение «Сортировка алертов SOC» (трек ML-инжиниринг).

Запуск: cd ML/app && poetry run python main.py  →  http://127.0.0.1:5050
"""
from course_model import configured_course

# В комплекте курса LightGBM и модель загружаются до pyarrow для совместимости с Windows.
configured_course()

from taipy.gui import Gui

from pages import advisor, inference, overview, shap_flow

if advisor.course_mode:
    from pages import course_xai as xai
else:
    from pages import xai

root = """
<|navbar|lov={nav}|>
"""
nav = [
    ("/overview", "Обзор потока"),
    ("/advisor", "Модель-советчик"),
    ("/inference", "Качество модели"),
    ("/xai", "Почему так?" if advisor.course_mode else "Объяснения (XAI)"),
    ("/shap-flow", "Объяснения на потоке"),
]

pages = {
    "/": root,
    "overview": overview.page,
    "advisor": advisor.page,
    "inference": inference.page,
    "xai": xai.page,
    "shap-flow": shap_flow.page,
}

if __name__ == "__main__":
    Gui(pages=pages, css_file="style.css").run(
        title="SOC · сортировка алертов", dark_mode=False, port=5050, use_reloader=False
    )
