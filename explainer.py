import sympy as sp
import re
from physics_db import NAMES_MAP


class PhysicsExplainer:
    def __init__(self):
        print("[Explainer] Интеллектуальный движок Axiom v6.5 (Квантовый рендеринг) готов.")

    # ФИКС ФОРМАТИРОВАНИЯ: Сверхмалые и сверхбольшие числа выводятся в красивом научном формате
    def _format_value(self, val):
        if isinstance(val, (int, float)):
            if val == 0:
                return "0.0"
            if abs(val) < 1e-3 or abs(val) > 1e6:
                return f"{val:.4e}"  # Экспоненциальный вид для микромира Иродова
            return f"{val:.4f}".rstrip('0').rstrip('.')
        return str(val)

    def _pretty_format(self, expr_str: str) -> str:
        res = expr_str
        res = res.replace("**2", "²").replace("*", "·")
        res = res.replace("P_gas", "P").replace("V_gas", "V").replace("T_gas", "T")
        res = res.replace("nu_moles", "ν").replace("R_gas", "R").replace("M_molar", "M")
        res = res.replace("U_energy", "U").replace("Q_heat", "Q").replace("A_gas", "A<sub>г</sub>")
        res = res.replace("eta_kpd", "η").replace("i_deg", "i")
        res = res.replace("k_spring", "k").replace("x_stretch", "x").replace("E_el", "E<sub>упр</sub>")
        res = res.replace("v_rms", "v<sub>кв</sub>").replace("gamma_adiab", "γ").replace("e_molek", "E₀")

        res = re.sub(r'\b([a-zA-Z]+)(\d+)\b', r'\1<sub>\2</sub>', res)
        return res

    def _find_bridge_symbols(self, equations):
        symbol_counts = {}
        for eq in equations:
            for s in eq.free_symbols:
                symbol_counts[s.name] = symbol_counts.get(s.name, 0) + 1
        bridges = [sym for sym, count in symbol_counts.items() if count >= 2 and sym not in ['g', 'R_gas', 'k_boltz']]
        return bridges

    def _generate_dynamic_commentary(self, eq, stage_idx, total_stages, bridges) -> str:
        symbols = [s.name for s in eq.free_symbols]

        active_bridges = [b for b in bridges if b in symbols]
        bridge_text = ""
        if active_bridges:
            bridge_names = [NAMES_MAP.get(b.lower(), b) for b in active_bridges]
            bridge_text = f"<br><span style='color: #38bdf8; font-size: 12px;'><b>💡 Стратегический мост:</b> Уравнение содержит ключевые связующие параметры ({', '.join(bridge_names)}), необходимые для объединения подсистем.</span>"

        if "v_rms" in symbols and "M_molar" in symbols:
            comment = "Применяем распределение Максвелла для средней квадратичной скорости молекул, увязывая макротемпературу с молярной массой газа."
        elif "e_molek" in symbols and "k_boltz" in symbols:
            comment = "Используем закон Больцмана для фиксации средней кинетической энергии теплового хаотического движения одной изолированной молекулы."
        elif "gamma_adiab" in symbols and "P1" in symbols:
            comment = "Разворачиваем уравнение Пуассона для адиабатического процесса (системы в теплоизолированной оболочке) для связи давлений и объемов."
        elif "k_spring" in symbols and "x_stretch" in symbols and "E_el" in symbols:
            comment = "Используем закон упругой деформации для расчета потенциальной энергии, запасенной сжатой или растянутой пружиной."
        elif "k_spring" in symbols and "x_stretch" in symbols and "F" in symbols:
            comment = "Задействуем закон Гука для нахождения силы упругости, возникающей при деформации конструкции."
        elif "P_gas" in symbols and "V_gas" in symbols and "nu_moles" in symbols:
            comment = "Применяем уравнение состояния идеального газа (Менделеева-Клапейрона) для фиксации макропараметров термодинамической системы."
        elif "Q_heat" in symbols and "U_energy" in symbols and "A_gas" in symbols:
            comment = "Используем первый закон термодинамики (закон сохранения энергии) для баланса переданной теплоты, работы и внутренней энергии."
        elif "eta_kpd" in symbols and "T1" in symbols and "T2" in symbols:
            comment = "Привлекаем термодинамический закон циклов Карно для связи эффективности машины с температурными уровнями среды."
        elif "F" in symbols and "m" in symbols and "a" in symbols:
            comment = "Разворачиваем второй закон Ньютона для описания динамики сил и темпа прироста линейной скорости тела."
        elif "mu" in symbols and "N" in symbols and "F_fr" in symbols:
            comment = "Задействуем закон Амонтона-Кулона для связывания силы трения скольжения с силой нормального давления поверхности."
        elif "s" in symbols and "t" in symbols and "a" in symbols:
            comment = "Интегрируем кинематическое уравнение перемещения для расчета пространственного положения точки во времени."
        else:
            parts_desc = [NAMES_MAP[s.lower()] for s in symbols if s.lower() in NAMES_MAP]
            comment = f"Уравнение физического баланса. Устанавливает строгую математическую зависимость между параметрами: {', '.join(parts_desc)}."

        prefix = f"<b>Шаг {stage_idx} из {total_stages}:</b> "
        context = "Формируем стартовый базис системы. " if stage_idx == 1 else (
            "Финальное замыкание графа. " if stage_idx == total_stages else "Промежуточная дедукция. ")

        return f"{prefix}{context}{comment}{bridge_text}"

    def generate_report(self, category: str, data: dict, target: str, result, actual_eqs=None):
        target_name = NAMES_MAP.get(target, target)
        used_equations = actual_eqs if actual_eqs else []
        total_stages = len(used_equations)
        bridges = self._find_bridge_symbols(used_equations)

        html = []
        html.append(
            "<div style='font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, sans-serif; color: #ffffff; line-height: 1.6;'>")
        html.append(
            "<h2 style='color: #3b82f6; border-bottom: 1px solid #334155; padding-bottom: 10px; text-align: left; font-size: 18px; font-weight: 600; letter-spacing: 0.5px;'>Axiom+ Нейросимволический Аналитический Отчет</h2>")

        html.append("<div style='margin-bottom: 20px;'>")
        html.append(
            f"<span style='color: #94a3b8; font-size: 11px; font-weight: 700; letter-spacing: 1px;'>[1. КЛАССИФИКАЦИЯ РАЗДЕЛА]</span><br>")
        html.append(f"<span style='font-size: 14px; color: #34d399;'><b>{category}</b></span></div>")

        html.append("<div style='margin-bottom: 20px;'>")
        html.append(
            "<span style='color: #94a3b8; font-size: 11px; font-weight: 700; letter-spacing: 1px;'>[2. ВЕКТОР ИСХОДНЫХ ПАРАМЕТРОВ (ДАНО)]</span><br>")
        html.append("<table style='width: 100%; margin-top: 5px; font-size: 13px; border-collapse: collapse;'>")
        for key, val in data.items():
            html.append(
                f"<tr style='border-bottom: 1px solid #1e293b;'><td style='padding: 4px 0; color: #cbd5e1;'>{NAMES_MAP.get(key, key)}</td><td style='text-align: right; color: #34d399; font-weight: 600;'>{self._pretty_format(key)} = {self._pretty_format(str(val))}</td></tr>")

        pretty_target = self._pretty_format(target)
        html.append(
            f"<tr><td style='padding: 6px 0; color: #ff9800; font-weight: bold;'>Искомый параметр</td><td style='text-align: right; color: #ff9800; font-weight: bold;'>{target_name} ({pretty_target})</td></tr>")
        html.append("</table></div>")

        html.append("<div style='margin-bottom: 20px;'>")
        html.append(
            "<span style='color: #94a3b8; font-size: 11px; font-weight: 700; letter-spacing: 1px;'>[3. ПОШАГОВЫЙ ДЕДУКТИВНЫЙ СИНТЕЗ РЕШЕНИЯ]</span><br>")

        for i, eq in enumerate(used_equations, 1):
            html.append(
                "<div style='background-color: #1e293b; padding: 12px; margin-top: 8px; border-radius: 6px; border-left: 4px solid #2563eb;'>")
            if isinstance(eq, sp.Eq):
                eq_str = f"{eq.lhs} = {eq.rhs}"
            else:
                args = eq.args
                eq_str = f"{args[0]} = {-args[1]}" if len(args) == 2 else f"{eq} = 0"

            html.append(
                f"<div style='color: #38bdf8; font-family: \"Courier New\", monospace; font-size: 15px; font-weight: bold; margin-bottom: 6px;'>{self._pretty_format(eq_str)}</div>")
            explanation = self._generate_dynamic_commentary(eq, i, total_stages, bridges)
            html.append(f"<div style='color: #94a3b8; font-size: 13px;'>{explanation}</div>")
            html.append("</div>")
        html.append("</div>")

        html.append(
            "<div style='background-color: #1e293b; padding: 15px; border-left: 4px solid #10b981; border-radius: 6px; margin-top: 20px;'>")
        html.append(
            "<span style='color: #94a3b8; font-size: 11px; font-weight: 700; letter-spacing: 1px;'>[4. МАТЕМАТИЧЕСКОЕ ЗАМЫКАНИЕ МАТРИЦЫ]</span><br>")

        if "Недостаточно" in str(result) or "Не удалось" in str(result):
            html.append(
                f"<div style='color: #f43f5e; font-size: 14px; font-weight: bold; margin-top: 5px;'>❌ Анализ приостановлен: {result}</div>")
        elif "Выражено аналитически" in str(result):
            raw_formula = str(result).split("=")[-1].strip()
            html.append(
                "<div style='color: #cbd5e1; font-size: 13px; margin-top: 5px;'>Символьное ядро SymPy провело редукцию промежуточных неопределенностей. Получена аналитическая модель:</div>")
            html.append(
                f"<div style='font-size: 18px; color: #10b981; font-weight: bold; margin-top: 5px;'>{self._pretty_format(target)} = {self._pretty_format(raw_formula)}</div>")
        else:
            html.append(
                "<div style='color: #cbd5e1; font-size: 13px; margin-top: 5px;'>Матрица уравнений успешно сомкнута. Выполнена подстановка констант и численный расчет:</div>")
            # ФИКС: Вызываем наш новый квантовый форматер для финального рендеринга числа
            html.append(
                f"<div style='font-size: 20px; color: #10b981; font-weight: bold; margin-top: 5px;'>{self._pretty_format(target)} = {self._format_value(result)}</div>")
        html.append("</div></div>")

        return "".join(html)