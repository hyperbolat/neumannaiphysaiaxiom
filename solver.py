import sympy as sp
import re
from physics_db import SYMBOLS_MAP, EQUATIONS, CONSTANTS, g, v0, t_ind, x_ind, R_res, R1_res, R2_res, m1, m2, T, F, a

SERIES_R_EQ = R_res - (R1_res + R2_res)
PARALLEL_R_EQ = 1 / R_res - (1 / R1_res + 1 / R2_res)

# "Динамика (Система тел)" содержит уравнения двух разных физических сценариев
# (машина Атвуда на гравитации vs связанные тела под внешней силой) — они
# взаимоисключающие для одного и того же T, отбираем нужный по данным задачи.
ATWOOD_EQS = {m1 * g - T - m1 * a, T - m2 * g - m2 * a}
EXTERNAL_FORCE_EQS = {F - (m1 + m2) * a, T - m2 * a}


class PhysicsSolver:
    def __init__(self):
        print("[Solver] Прецизионный каскад решений Axiom v7.2 развернут.")

    @staticmethod
    def _connected_subsystem(equations, target_sym):
        """
        Оставляет только уравнения, транзитивно связанные с целевым символом
        через общие переменные (обход в ширину по графу "уравнение - символ").
        Уравнения из случайно подмешанных категорий (см. кросс-лоадинг),
        которые физически никак не связаны с искомой величиной, вносят в
        систему свободные неизвестные и ломают sp.solve на пустом месте —
        эта фильтрация устраняет проблему в общем виде, а не для конкретной
        пары категорий.
        """
        frontier = {target_sym}
        relevant = []
        changed = True
        while changed:
            changed = False
            for eq in equations:
                if eq in relevant:
                    continue
                if eq.free_symbols & frontier:
                    relevant.append(eq)
                    new_syms = eq.free_symbols - frontier
                    if new_syms:
                        frontier |= new_syms
                        changed = True
        return relevant

    @staticmethod
    def _solve_cascade(equations, target_symbol, target_param):
        """
        Прогоняет каскад из 3 попыток sp.solve и возвращает (best_numeric,
        best_symbolic) — оба могут быть None, если решение не найдено.
        Вынесено в отдельный метод, чтобы можно было сначала попробовать
        маленькую "достаточную" систему, и только при неудаче — большую.
        """
        unknown_symbols = list(set().union(*[eq.free_symbols for eq in equations]) - {target_symbol}) \
            if equations else []

        attempts = []
        sol1 = sp.solve(equations, [target_symbol], dict=True)
        if sol1:
            attempts.append(sol1)

        sol2 = sp.solve(equations, unknown_symbols + [target_symbol], dict=True)
        if sol2:
            attempts.append(sol2)

        sol3 = sp.solve(equations, dict=True)
        if sol3:
            attempts.append(sol3)

        best_numeric, best_symbolic = None, None
        for solutions in attempts:
            for sol in solutions:
                if target_symbol not in sol:
                    continue
                val = sol[target_symbol]
                if val.is_real and len(val.free_symbols) == 0:
                    if float(val) > 0 or target_param in ['v', 'v_gas', 'u', 'a', 'e_molek']:
                        if best_numeric is None:
                            best_numeric = float(val)
                elif len(val.free_symbols) > 0 and best_symbolic is None:
                    best_symbolic = val
            if best_numeric is not None:
                break
        return best_numeric, best_symbolic

    def calculate(self, data: dict, target_param: str, category: str, task_text: str = ""):
        local_data = data.copy()
        if target_param in local_data:
            del local_data[target_param]

        is_calculus_task = ("высшая математика" in category.lower()) or any(
            isinstance(val, str) and ('t' in val or 'x' in val) for val in local_data.values()
        )
        if is_calculus_task:
            return self._solve_calculus(local_data, SYMBOLS_MAP.get(target_param), target_param), []

        requested_categories = [cat.strip() for cat in re.split(r'[,;+]', category)]
        extended_categories = set(requested_categories)
        # ВАЖНО: target_param намеренно НЕ подмешивается в общий набор ключей для
        # кросс-лоадинга. Короткие имена (f, u, m, a...) физически переиспользуются
        # в разных разделах (сила есть и в динамике, и в электростатике), и если
        # искомая переменная называется, скажем, 'f', это ложно триггерило бы
        # соседний раздел просто по совпадению имени, засоряя систему чужими
        # неизвестными. 'alpha' и 'u' — safe legacy-исключения, добавлены точечно.
        all_task_keys = set(local_data.keys())
        if target_param in ('alpha', 'u'):
            all_task_keys = all_task_keys | {target_param}
        text_lower = task_text.lower()

        # Кросс-лоадинг разделов
        if any(k in all_task_keys for k in ['f', 'm', 'f_fr', 'n', 'mu']): extended_categories.add(
            "Динамика (Одно тело)")
        if any(k in all_task_keys for k in ['m1', 'm2', 't_tension']): extended_categories.add("Динамика (Система тел)")
        if any(k in all_task_keys for k in ['s', 'v', 'v0', 'a', 'v1', 'v2']): extended_categories.add("Кинематика")
        if any(k in all_task_keys for k in ['m_torque', 'i_inert', 'eps', 'omega', 'r_rad']): extended_categories.add(
            "Динамика вращательного движения")
        if any(k in all_task_keys for k in ['p', 'ek', 'ep', 'a_work', 'p_pow', 'u']) or any(
            w in text_lower for w in ["столкн", "удар"]): extended_categories.add("Законы сохранения")
        if any(k in all_task_keys for k in
               ['p_gas', 'v_gas', 't_gas', 'nu_moles', 'q_heat', 'u_energy', 'a_gas', 'eta_kpd', 't1', 't2', 'v1_vol',
                'v2_vol', 'p1_pres', 'p2_pres']):
            extended_categories.add("Молекулярная физика и Термодинамика")
        if any(k in all_task_keys for k in
               ['q1_ch', 'q2_ch', 'r_dist', 'e_field', 'c_cap', 'w_cap', 'k_coulomb']):
            extended_categories.add("Электростатика")
        if any(k in all_task_keys for k in
               ['i_current', 'r_res', 'r1_res', 'r2_res', 'p_el', 'q_joule']) or any(
            w in text_lower for w in ["сопротивлен", "цепь", "ампер", "вольт"]):
            extended_categories.add("Постоянный ток")
        if any(k in all_task_keys for k in
               ['b_field', 'f_lor', 'f_amp', 'l_wire', 's_area', 'phi_flux', 'emf_ind']) or any(
            w in text_lower for w in ["магнитн", "индукц"]):
            extended_categories.add("Магнетизм и электромагнитная индукция")
        if any(k in all_task_keys for k in
               ['t_period', 'freq', 'a_ampl', 'l_pend', 'w_osc']) or any(
            w in text_lower for w in ["маятник", "колебан"]):
            extended_categories.add("Колебания")
        if any(k in all_task_keys for k in ['v_wave', 'lambda_wave']) or "волн" in text_lower:
            extended_categories.add("Волны")
        if any(k in all_task_keys for k in
               ['f_lens', 'd_obj', 'd_img', 'gamma_magnif', 'n1_refr', 'n2_refr', 'angle_i', 'angle_r']) or any(
            w in text_lower for w in ["линз", "преломлен", "изображени"]):
            extended_categories.add("Оптика")

        # Геометрический арбитраж
        if 'alpha' in all_task_keys or any(w in text_lower for w in ["наклон", "угол", "соскальз"]):
            extended_categories.add("Динамика (Наклонная плоскость)")
            extended_categories.discard("Динамика (Одно тело)")

        # Если явно запрошен немеханический раздел (электричество, оптика,
        # колебания...), мехaника (Кинематика/Динамика), подмешанная только
        # эвристикой по общим именам (v, t, m, F переиспользуются везде),
        # почти наверняка ложная — она не была в исходной классификации и
        # только раздувает систему. Оставляем механику, только если она
        # была запрошена явно.
        NON_MECHANICS = {"Электростатика", "Постоянный ток", "Магнетизм и электромагнитная индукция",
                          "Колебания", "Волны", "Оптика", "Молекулярная физика и Термодинамика",
                          "Законы сохранения"}
        MECHANICS_AUTO = {"Кинематика", "Динамика (Одно тело)", "Динамика (Система тел)",
                           "Динамика (Наклонная плоскость)", "Динамика вращательного движения"}
        if extended_categories & NON_MECHANICS:
            for auto_cat in MECHANICS_AUTO - set(requested_categories):
                extended_categories.discard(auto_cat)

        if any(w in text_lower for w in ["столкн", "удар"]) or "u" in all_task_keys:
            if "v2" not in local_data and "v1" in local_data: local_data["v2"] = 0.0

        base_template_equations = []
        for cat in extended_categories:
            if cat in EQUATIONS: base_template_equations.extend(EQUATIONS[cat])

        # Формулы последовательного и параллельного соединения резисторов —
        # взаимоисключающие модели одной и той же цепи. Обе сразу дают
        # противоречивую систему для одного и того же R_res. Если R1/R2 не
        # заданы и не являются целью — убираем обе. Если заданы — оставляем
        # только ту, что соответствует формулировке задачи (по умолчанию,
        # при отсутствии явного указания, оставляем последовательную как
        # более частый случай в школьных задачах).
        if 'r1_res' not in local_data and 'r2_res' not in local_data and target_param not in ('r1_res', 'r2_res'):
            base_template_equations = [
                eq for eq in base_template_equations
                if eq != SERIES_R_EQ and eq != PARALLEL_R_EQ
            ]
        elif any(w in text_lower for w in ["параллельн"]):
            base_template_equations = [eq for eq in base_template_equations if eq != SERIES_R_EQ]
        else:
            base_template_equations = [eq for eq in base_template_equations if eq != PARALLEL_R_EQ]

        # Машина Атвуда (движение за счёт гравитации) vs связанные тела под
        # внешней силой — тоже взаимоисключающие модели одной и той же
        # системы (обе используют T). Если сила F дана или является целью —
        # это явно "внешняя сила" сценарий; иначе — гравитационный Атвуд.
        if 'f' in local_data or target_param == 'f':
            base_template_equations = [eq for eq in base_template_equations if eq not in ATWOOD_EQS]
        else:
            base_template_equations = [eq for eq in base_template_equations if eq not in EXTERNAL_FORCE_EQS]

        detected_entities = set()
        for key in local_data.keys():
            if key == 'v0':
                continue
            match = re.search(r'\d+$', str(key))
            if match: detected_entities.add(match.group())

        # Клонирование по сущностям (m1/m2, v1/v2...) — инвазивная трансформация:
        # она переименовывает символы вроде 'v' в 'v1' по всей категории. Если
        # сработает ошибочно (например, экстрактор случайно пометил число как
        # v1, хотя пары v2/m2 в задаче нет), это может незаметно "стереть" саму
        # искомую переменную из системы. Настоящие двухтельные задачи всегда
        # дают ПАРУ индексов (m1 и m2, или v1 и v2) — поэтому включаем
        # клонирование только когда индексов действительно >= 2, а не по
        # одному случайному совпадению.
        if len(detected_entities) < 2:
            detected_entities = set()

        final_system_equations = []

        def get_indexed_symbol(base_sym, index):
            return sp.Symbol(f"{base_sym.name}{index}")

        for eq in base_template_equations:
            if isinstance(eq, sp.Symbol) and eq.name == "calculus_flag": continue
            contains_system_indices = any(
                s.name in ["m1", "m2", "v1", "v2", "u", "T", "T1", "T2", "V1", "V2", "P1", "P2"] for s in
                eq.free_symbols)
            if detected_entities and not contains_system_indices:
                for entity_idx in detected_entities:
                    substitutions = {}
                    for sym in eq.free_symbols:
                        if sym.name in ["v", "v0", "s", "a", "F", "m", "F_fr", "N", "Ek", "Ep", "p", "t"]:
                            substitutions[sym] = get_indexed_symbol(sym, entity_idx)
                    cloned_eq = eq.subs(substitutions)
                    if cloned_eq != 0 and cloned_eq not in final_system_equations: final_system_equations.append(
                        cloned_eq)
            else:
                if eq not in final_system_equations: final_system_equations.append(eq)

        is_translational_kinematics = any(
            cat in ["Кинематика", "Динамика (Одно тело)", "Динамика (Система тел)", "Динамика (Наклонная плоскость)"]
            for cat in extended_categories)

        coupling_laws = []
        if is_translational_kinematics:
            has_dynamic_triggers = any(
                k in local_data for k in ['f', 'm', 'f1', 'm1', 'f2', 'm2', 't_tension', 'f_fr', 'mu', 'alpha'])
            has_dynamic_words = any(w in text_lower for w in
                                    ["сила", "масса", "толкали", "трение", "трения", "плоскость", "наклонной",
                                     "ускорен"])

            if not has_dynamic_triggers and not has_dynamic_words:
                if "ускорен" not in text_lower and "a" not in local_data and "a1" not in local_data and "a2" not in local_data:
                    if detected_entities:
                        for entity_idx in detected_entities:
                            coupling_laws.append(sp.Eq(get_indexed_symbol(sp.Symbol('a'), entity_idx), 0))
                            coupling_laws.append(sp.Eq(get_indexed_symbol(sp.Symbol('v0'), entity_idx),
                                                       get_indexed_symbol(sp.Symbol('v'), entity_idx)))
                    else:
                        coupling_laws.append(sp.Eq(SYMBOLS_MAP['a'], 0))
                        coupling_laws.append(sp.Eq(SYMBOLS_MAP['v0'], SYMBOLS_MAP['v']))

            if detected_entities:
                for entity_idx in detected_entities: coupling_laws.append(
                    sp.Eq(get_indexed_symbol(sp.Symbol('t'), entity_idx), sp.Symbol('t')))
                entities_list = sorted(list(detected_entities))
                if len(entities_list) >= 2:
                    s1 = get_indexed_symbol(sp.Symbol('s'), entities_list[0])
                    s2 = get_indexed_symbol(sp.Symbol('s'), entities_list[1])
                    if any(w in text_lower for w in ["навстречу", "из а в б"]):
                        coupling_laws.append(sp.Eq(s1 + s2, sp.Symbol('s')))
                    elif any(w in text_lower for w in ["догнал", "попутно", "вдогонку"]):
                        coupling_laws.append(sp.Eq(s1, s2))

        final_system_equations.extend(coupling_laws)
        graph_equations = final_system_equations.copy()

        subs_dict = {}
        for key, value in local_data.items():
            if key in SYMBOLS_MAP:
                sym_obj = SYMBOLS_MAP[key]
            else:
                base_part = re.sub(r'\d+$', '', key)
                idx_part = re.search(r'\d+$', key)
                if base_part in SYMBOLS_MAP and idx_part:
                    sym_obj = get_indexed_symbol(SYMBOLS_MAP[base_part], idx_part.group())
                else:
                    sym_obj = sp.Symbol(key)
            if not isinstance(value, str): subs_dict[sym_obj] = value

        if target_param in SYMBOLS_MAP:
            target_symbol = SYMBOLS_MAP[target_param]
        else:
            base_part = re.sub(r'\d+$', '', target_param)
            idx_part = re.search(r'\d+$', target_param)
            if base_part in SYMBOLS_MAP and idx_part:
                target_symbol = get_indexed_symbol(SYMBOLS_MAP[base_part], idx_part.group())
            else:
                target_symbol = sp.Symbol(target_param)

        for c_sym, c_val in CONSTANTS.items():
            if target_symbol != c_sym: subs_dict[c_sym] = c_val

        if "v0" not in local_data and target_symbol != v0 and is_translational_kinematics: subs_dict[v0] = 0.0

        # =========================================================================
        # УРОВЕНЬ 0: ЖАДНОЕ ПОСТРОЕНИЕ МИНИМАЛЬНО ДОСТАТОЧНОЙ ЦЕПОЧКИ
        # =========================================================================
        # Некоторые категории (особенно термодинамика — 12 уравнений в одной
        # куче) настолько плотно связаны общими символами (R_gas, T_gas...),
        # что обрезка по связности ничего не обрезает — все уравнения реально
        # транзитивно достижимы. Но для конкретной задачи обычно нужно 1-3
        # уравнения, а не все 12.
        #
        # Строим цепочку так же, как решал бы человек: повторно берём любое
        # уравнение, которое вводит НЕ БОЛЬШЕ ОДНОЙ новой (ещё неизвестной)
        # переменной сверх уже известного — эта переменная тем самым сама
        # становится "известной" для следующего шага. Продолжаем, пока можно
        # находить такие уравнения. Пример (Карно): T1,T2 известны — уравнение
        # eta=(T1-T2)/T1 вводит ровно одну новую (eta) -> берём, eta становится
        # известной; дальше eta=A/Q вводит ровно одну новую (A=цель) -> берём.
        known_symbols = set(subs_dict.keys())
        minimal_equations = []
        remaining = list(final_system_equations)
        changed = True
        while changed and target_symbol not in known_symbols:
            changed = False
            for eq in remaining[:]:
                new_unknowns = eq.free_symbols - known_symbols
                if len(new_unknowns) <= 1:
                    minimal_equations.append(eq)
                    remaining.remove(eq)
                    known_symbols |= new_unknowns
                    changed = True

        if minimal_equations and target_symbol in known_symbols:
            substituted_minimal = [eq.subs(subs_dict) for eq in minimal_equations]
            best_numeric, best_symbolic = self._solve_cascade(substituted_minimal, target_symbol, target_param)
            if best_numeric is not None:
                return best_numeric, minimal_equations
            if best_symbolic is not None:
                clean_formula = str(sp.simplify(best_symbolic))
                return f"Выражено аналитически: {target_param} = {clean_formula}", minimal_equations

        # =========================================================================
        # УРОВЕНЬ 1: ПОЛНАЯ СИСТЕМА, ОБРЕЗАННАЯ ПО СВЯЗНОСТИ С ЦЕЛЬЮ
        # =========================================================================
        # Если минимальной системы не хватило (обычно — многошаговые задачи,
        # где ответ выводится через промежуточные величины), берём подграф,
        # транзитивно связанный с целью. Это и чинит побочные эффекты
        # слишком щедрого кросс-лоадинга категорий, и делает отчёт для
        # пользователя чище (не показываем уравнения не по теме).
        pruned_equations = self._connected_subsystem(final_system_equations, target_symbol)
        if not pruned_equations:
            # Цель не встречается НИ В ОДНОМ уравнении системы — чаще всего
            # значит, что извлечение данных перепутало метку (например,
            # искомая переменная случайно "переименовалась" при клонировании
            # по сущностям, см. entity-cloning выше). Откат на полный
            # несокращённый список тут не поможет: раз цели нет ни в одном
            # уравнении, её не будет и в объединении всех уравнений — только
            # зря раздуваем систему и прячем реальную причину в отчёте.
            return ("Не удалось замкнуть матрицу уравнений (искомая величина "
                    f"'{target_param}' не встречается ни в одном подходящем уравнении — "
                    f"проверьте, не перепутаны ли извлечённые данные)."), []
        final_system_equations = pruned_equations
        graph_equations = final_system_equations.copy()

        substituted_system = [eq.subs(subs_dict) for eq in final_system_equations]

        best_numeric, best_symbolic = self._solve_cascade(substituted_system, target_symbol, target_param)
        if best_numeric is not None:
            return best_numeric, graph_equations
        if best_symbolic is not None:
            clean_formula = str(sp.simplify(best_symbolic))
            return f"Выражено аналитически: {target_param} = {clean_formula}", graph_equations

        return "Не удалось замкнуть матрицу уравнений.", graph_equations

    def _solve_calculus(self, local_data: dict, target_symbol, target_param: str):
        try:
            func_expr = None
            for k, val in local_data.items():
                if isinstance(val, str):
                    clean_val = val.lower().replace("тау", "tau")
                    func_expr = sp.sympify(clean_val, locals={'t': t_ind, 'x': x_ind, 'v0': SYMBOLS_MAP['v0'],
                                                              'tau': SYMBOLS_MAP['tau']})
                    break
            if target_param == 's':
                t_solutions = sp.solve(func_expr, t_ind)
                t_stop = t_solutions[0] if t_solutions else t_ind
                path_expr = sp.integrate(func_expr, (t_ind, 0, t_stop))
                return f"Выражено через определенный интеграл высшей школы:<br> s = ∫ v(t)dt от 0 до {t_stop} = <b>{sp.simplify(path_expr)}</b>"
            return "Calculus-паттерн еще не описан."
        except Exception as e:
            return f"Ошибка Calculus: {str(e)}"