"""Taipy-приложение «Сортировка алертов SOC» (трек ML-инжиниринг).

Запуск: cd ML/app && poetry run python main.py  →  http://127.0.0.1:5050
"""
from taipy.gui import Gui

from pages import advisor, overview

root = """
<|navbar|lov={nav}|>
"""
nav = [("/overview", "Обзор потока"), ("/advisor", "Модель-советчик")]

pages = {
    "/": root,
    "overview": overview.page,
    "advisor": advisor.page,
}

if __name__ == "__main__":
    Gui(pages=pages, css_file="style.css").run(
        title="SOC · сортировка алертов", dark_mode=False, port=5050, use_reloader=False
    )
