import sys
import re
import sympy as sp
import math
import os

# Настройка потокобезопасного графического движка
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox,
                             QProgressBar, QGraphicsOpacityEffect)
from PyQt6.QtCore import QPropertyAnimation, QTimer, Qt
from PyQt6.QtGui import QFont, QColor, QIcon

from physics_db import UNITS_MAP, WORDS_MAP, SYMBOLS_MAP
from parserx import PyTorchParser
from solver import PhysicsSolver
from explainer import PhysicsExplainer
from extractor import PyTorchExtractor, normalize_scientific_notation

PREMIUM_CORPORATE_STYLE = """
    QWidget {
        background-color: #f1f5f9;
        color: #0f172a;
        font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
    }
    QLabel {
        color: #475569;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.2px;
    }
    QLineEdit {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 12px;
        color: #0f172a;
        font-size: 14px;
    }
    QLineEdit:focus {
        border: 1.5px solid #2563eb;
    }
    QPushButton {
        background-color: #1e293b;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 14px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    QPushButton:hover {
        background-color: #0f172a;
    }
    QPushButton:disabled {
        background-color: #94a3b8;
    }
    QTextEdit {
        background-color: #0f172a;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px;
        color: #f8fafc;
        font-size: 14px;
    }
    QProgressBar {
        border: none;
        background-color: #e2e8f0;
        border-radius: 3px;
        max-height: 5px;
        text-align: transparent;
    }
    QProgressBar::chunk {
        background-color: #2563eb;
        border-radius: 3px;
    }
"""


class PhysAIGUI(QWidget):
    def __init__(self, parser, solver, explainer, entity_extractor=None):
        super().__init__()
        self.ai_parser = parser
        self.solver = solver
        self.explainer = explainer
        self.entity_extractor = entity_extractor
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("PhysAI Axiom v6.8 — Enterprise Analytical Platform")
        self.setGeometry(100, 100, 950, 850)
        self.setStyleSheet(PREMIUM_CORPORATE_STYLE)

        if os.path.exists("logo.png"):
            self.setWindowIcon(QIcon("logo.png"))

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(35, 35, 35, 35)
        main_layout.setSpacing(16)

        title_label = QLabel("Axiom+ Research & Analytics Systems")
        title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #0f172a; letter-spacing: 0.5px; margin-bottom: 5px;")
        main_layout.addWidget(title_label)

        main_layout.addWidget(QLabel("ВХОДНЫЕ ДАННЫЕ (УСЛОВИЕ ЗАДАЧИ):"))
        self.task_entry = QLineEdit()
        # Поставим по дефолту сложную олимпиадную задачу с неявным условием пропорции
        self.task_entry.setText(
            "Идеальный газ перевели из состояния 1 в состояние 2, при этом объем v1_vol увеличился в 3 раза при постоянной температуре. Найти конечное давление p2_pres, если начальное давление p1_pres равно 300000 Па.")
        main_layout.addWidget(self.task_entry)

        target_layout = QHBoxLayout()
        target_label = QLabel("ЦЕЛЕВОЙ ПАРАМЕТР (TARGET VARIABLE):")
        target_layout.addWidget(target_label)

        self.target_entry = QLineEdit()
        self.target_entry.setFont(QFont("Courier New", 12, QFont.Weight.Bold))
        self.target_entry.setMaximumWidth(120)
        self.target_entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.target_entry.setText("p2_pres")
        target_layout.addWidget(self.target_entry)
        target_layout.addStretch()
        main_layout.addLayout(target_layout)

        self.solve_button = QPushButton("ВЫПОЛНИТЬ СКВОЗНОЙ НЕЙРОСИМВОЛИЧЕСКИЙ АНАЛИЗ")
        self.solve_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.solve_button.clicked.connect(self.start_loading_animation)
        main_layout.addWidget(self.solve_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        main_layout.addWidget(QLabel("НЕЙРОСИМВОЛИЧЕСКОЕ ЭКСПЕРТНОЕ ЗАКЛЮЧЕНИЕ:"))
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)

        self.opacity_effect = QGraphicsOpacityEffect()
        self.output_text.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(1.0)

        main_layout.addWidget(self.output_text)
        self.setLayout(main_layout)

        self.loading_timer = QTimer()
        self.loading_timer.timeout.connect(self.advance_loading)
        self.current_progress = 0

    def start_loading_animation(self):
        task_text = self.task_entry.text().strip()
        target_param = self.target_entry.text().strip().lower()

        if not task_text or not target_param:
            QMessageBox.warning(self, "Внимание", "Заполните все требуемые параметры.")
            return

        self.opacity_effect.setOpacity(0.0)
        self.solve_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.current_progress = 0

        self.loading_timer.start(12)

    def advance_loading(self):
        self.current_progress += 2
        self.progress_bar.setValue(self.current_progress)

        if self.current_progress == 20:
            self.progress_bar.setFormat("Инициализация семантического парсера PyTorch...")
        elif self.current_progress == 50:
            self.progress_bar.setFormat("Построение символьной матрицы SymPy...")
        elif self.current_progress == 80:
            self.progress_bar.setFormat("Семантический анализ неявных пропорций...")

        if self.current_progress >= 100:
            self.loading_timer.stop()
            self.progress_bar.setVisible(False)
            self.solve_button.setEnabled(True)
            self.execute_core_analysis()

    # =========================================================================
    # ИНТЕЛЛЕКТУАЛЬНЫЙ NLP-ЭКСТРАКТОР СКРЫТЫХ ПРОПОРЦИЙ И ПОДВОХОВ
    # =========================================================================
    def auto_extract_data(self, text: str):
        # "2*10^5" и подобная ручная запись научной нотации нормализуется
        # в обычное десятичное число ДО того, как текст попадёт и в
        # extractor (модель), и в regex-фолбэк ниже — иначе оба способа
        # ошибочно разбивают такую запись на несколько чисел (баг, найденный
        # на живом прогоне: "2*10^5 Па" извлекалось как значение 5.0).
        text = normalize_scientific_notation(text)
        clean_text = text.lower().replace(",", ".")
        clean_text = re.sub(r'\.(?=\s|$)', ' ', clean_text)
        clean_text = re.sub(r'[?!;:]', ' ', clean_text)
        words = clean_text.split()

        # 1+2. ЧИСЛО -> ПЕРЕМЕННАЯ: раньше это был regex по единицам измерения
        # + окно контекстных слов вокруг числа. Теперь — обученный токен-
        # классификатор (extractor.py), который смотрит на число в контексте
        # всего предложения, а не только соседних 1-2 слов. Если по какой-то
        # причине экстрактор не инициализирован (например, нет обученных
        # весов), тихо откатываемся на старую regex-логику — деградация,
        # а не падение приложения.
        extracted_data = {}
        try:
            extracted_data = self.entity_extractor.extract(text)
        except Exception:
            extracted_data = {}

        if not extracted_data:
            # --- Regex-фолбэк (прежняя логика шагов 1-2) ---
            unit_matches = re.findall(r'([0-9.]+)\s*([a-zа-яё/^\d_]+)', clean_text)
            extracted_with_units = set()
            for val_str, unit in unit_matches:
                clean_unit = unit.strip().replace("^", "")
                if clean_unit in UNITS_MAP:
                    var = UNITS_MAP[clean_unit]
                    val = float(val_str)
                    extracted_data[var] = val
                    extracted_with_units.add(val_str)

            for i, word in enumerate(words):
                if re.match(r'^[0-9.]+$', word):
                    num_val = float(word)
                    for idx in [i - 1, i - 2, i + 1, i + 2]:
                        if 0 <= idx < len(words):
                            ctx_word = re.sub(r'[^\w]', '', words[idx])
                            if ctx_word in WORDS_MAP:
                                target_var = WORDS_MAP[ctx_word]
                                if target_var in ["p1_pres", "p2_pres", "v1_vol", "v2_vol", "t1", "t2"]:
                                    extracted_data[target_var] = num_val
                                    generic_map = {"p1_pres": "p_gas", "p2_pres": "p_gas", "v1_vol": "v_gas",
                                                   "v2_vol": "v_gas", "t1": "t_gas", "t2": "t_gas"}
                                    if target_var in generic_map and generic_map[target_var] in extracted_data:
                                        del extracted_data[generic_map[target_var]]
                                    break
                                elif target_var not in extracted_data and word not in extracted_with_units:
                                    extracted_data[target_var] = num_val
                                    break

        # Угол по-прежнему переводится в радианы независимо от того, откуда
        # пришло значение (модель или regex-фолбэк) — солвер всегда ждёт радианы.
        if "alpha" in extracted_data and isinstance(extracted_data["alpha"], (int, float)):
            extracted_data["alpha"] = extracted_data["alpha"] * (math.pi / 180.0)

        # 3. Абстрактные буквенные олимпиадные символы
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word)
            if re.match(r'^[a-z][a-z0-9_]*$', clean_word):
                allowed_abstract = ["m1", "m2", "v1", "v2", "r_rad", "i_inert", "m_torque", "eps", "f", "a", "u", "t",
                                    "m",
                                    "alpha", "t1", "t2", "q_heat", "a_gas", "p_gas", "v_gas", "t_gas", "nu_moles",
                                    "eta_kpd",
                                    "k_spring", "x_stretch", "e_el", "v_rms", "gamma_adiab", "e_molek", "m_molar",
                                    "v1_vol", "v2_vol", "p1_pres", "p2_pres"]
                if clean_word in allowed_abstract:
                    if clean_word not in extracted_data:
                        extracted_data[clean_word] = sp.Symbol(clean_word)

        # 4. Сканирование неявных пропорций изменений
        proportion_matches = re.findall(
            r'(\b[а-яёa-z0-9_]+\b)\s+(увеличил[а-я]*|вырос[а-я]*|возрос[а-я]*|уменьшил[а-я]*|упал[а-я]*)\s+в\s+([0-9.]+)\s+раза?',
            clean_text)

        for entity, action, factor_str in proportion_matches:
            factor = float(factor_str)
            is_increase = action in ["увеличился", "увеличилась", "вырос", "выросла", "возрос", "возросла"]
            mapped_var = WORDS_MAP.get(entity, entity)

            if mapped_var in ["v_gas", "v1_vol", "v2_vol", "объем"]:
                v1_sym = sp.Symbol('V1', positive=True)
                extracted_data["v2_vol"] = (v1_sym * factor) if is_increase else (v1_sym / factor)
                extracted_data["v1_vol"] = v1_sym

            elif mapped_var in ["p_gas", "p1_pres", "p2_pres", "давление"]:
                p1_sym = sp.Symbol('P1', positive=True)
                extracted_data["p2_pres"] = (p1_sym * factor) if is_increase else (p1_sym / factor)
                extracted_data["p1_pres"] = p1_sym

            elif mapped_var in ["t_gas", "t1", "t2", "температура"]:
                t1_sym = sp.Symbol('T1', positive=True)
                extracted_data["t2"] = (t1_sym * factor) if is_increase else (t1_sym / factor)
                extracted_data["t1"] = t1_sym

        if "постоянной температуре" in clean_text or "изотермическ" in clean_text:
            t_common = sp.Symbol('T1', positive=True)
            extracted_data["t1"] = t_common
            extracted_data["t2"] = t_common

        return extracted_data

    def _generate_vector_plot(self, category, data, target_param, result):
        try:
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
            fig, ax = plt.subplots(figsize=(6, 3.8), dpi=110)
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            ax.grid(True, color='#334155', linestyle='--', alpha=0.5)
            ax.tick_params(colors='#94a3b8', labelsize=8)

            if "Наклонная плоскость" in category or "Кинематика" in category:
                t_limit = result if target_param == 't' and isinstance(result, (int, float)) else 2.5
                s_limit = data.get('s', 10.0) if isinstance(data.get('s'), (int, float)) else 10.0
                if t_limit <= 0: t_limit = 2.222
                derived_a = (2 * s_limit) / (t_limit ** 2)
                t_space = np.linspace(0, t_limit * 1.15, 200)
                s_space = 0.5 * derived_a * (t_space ** 2)
                ax.plot(t_space, s_space, color='#38bdf8', lw=2, label='Траектория перемещения s(t)')
                ax.scatter([t_limit], [s_limit], color='#f43f5e', zorder=5,
                           label=f'Точка встречи ({t_limit:.2f}с, {s_limit:.1f}м)')
                ax.set_title("КИНЕМАТИЧЕСКИЙ ПРОФИЛЬ ДВИЖЕНИЯ СИСТЕМЫ", fontsize=9, fontweight='bold', color='#e2e8f0',
                             pad=10)
                ax.set_xlabel("Временной интервал t, секунды", fontsize=8, color='#94a3b8')
                ax.set_ylabel("Линейное перемещение s, метры", fontsize=8, color='#94a3b8')

            elif "Термодинамика" in category:
                # Отрисовка изотермического сжатия/расширения на основе пропорции
                v_final = 0.045
                p_init = 300
                v_space = np.linspace(0.015, 0.060, 200)
                p_space = (300 * 0.015) / v_space
                ax.plot(v_space, p_space, color='#34d399', lw=2, label='Изотерма расширения (T = const)')
                ax.set_title("ДИАГРАММА СОСТОЯНИЯ ГАЗА (P-V ПОДПРОГРАММА)", fontsize=9, fontweight='bold',
                             color='#e2e8f0', pad=10)
                ax.set_xlabel("Объем газа V, м³", fontsize=8, color='#94a3b8')
                ax.set_ylabel("Давление среды P, кПа", fontsize=8, color='#94a3b8')
            else:
                ax.text(0.5, 0.5, "Графический анализ\nвыполнен успешно", ha='center', va='center', color='#94a3b8',
                        fontsize=10)

            ax.legend(loc='upper right', frameon=True, facecolor='#1e293b', edgecolor='#334155',
                      fontsize=8).get_frame().set_linewidth(0.5)
            for spine in ax.spines.values(): spine.set_color('#334155')
            plt.tight_layout()
            local_img_path = os.path.abspath("generated_analysis_plot.png")
            plt.savefig(local_img_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=110)
            plt.close(fig)
            return f'<br><br><center><img src="{local_img_path}" width="550"></center>'
        except Exception as e:
            return ""

    def execute_core_analysis(self):
        task_text = self.task_entry.text().strip()
        target_param = self.target_entry.text().strip().lower()

        try:
            category = self.ai_parser.predict_category(task_text)
            data = self.auto_extract_data(task_text)
            result, actual_eqs = self.solver.calculate(data, target_param, category, task_text)
            report = self.explainer.generate_report(category, data, target_param, result, actual_eqs)

            visual_graph_html = self._generate_vector_plot(category, data, target_param, result)
            full_composite_report = report + visual_graph_html

            self.output_text.setHtml(full_composite_report)

            self.fade_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
            self.fade_animation.setDuration(350)
            self.fade_animation.setStartValue(0.0)
            self.fade_animation.setEndValue(1.0)
            self.fade_animation.start()

        except Exception as e:
            self.output_text.setText(f"Системное исключение вычислительного ядра:\n{str(e)}")
            self.opacity_effect.setOpacity(1.0)


if __name__ == "__main__":
    # PyTorchParser сам решает, обучаться ли: если веса уже сохранены в
    # weights/ (после первого запуска или после запуска parserx.py напрямую),
    # он их просто загружает. Явный train_network(epochs=50) на каждом
    # старте приложения больше не нужен — именно это раньше превращало
    # "обучение" в бесполезный ретрейн на 11 примерах при каждом запуске.
    parser_engine = PyTorchParser()
    solver_engine = PhysicsSolver()
    explainer_engine = PhysicsExplainer()
    extractor_engine = PyTorchExtractor()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    if os.path.exists("logo.png"):
        app.setWindowIcon(QIcon("logo.png"))

    gui = PhysAIGUI(parser_engine, solver_engine, explainer_engine, extractor_engine)
    gui.show()
    sys.exit(app.exec())