# -*- coding: utf-8 -*-
"""Mini NPU Simulator

- 모드 1: 사용자 입력(3x3) 필터 A/B + 패턴 -> MAC 연산 -> 판정
- 모드 2: data.json 일괄 분석 (라벨 정규화 / 스키마 검증 / PASS-FAIL / 성능 분석 / 결과 요약)
- 모드 3: 보너스 (패턴 생성기 + 1차원 배열 최적화 성능 비교)

제약: Python 3.8 이상, 표준 라이브러리(json, time, os, re, math)만 사용.
MAC(Multiply-Accumulate) 연산은 반복문으로 직접 구현한다.
"""

import json
import math
import os
import re
import time

# ----------------------------------------------------------------------
# 상수
# ----------------------------------------------------------------------
EPSILON = 1e-9          # 점수 비교 허용오차 (|a - b| < EPSILON 이면 동점)
REPEAT = 10             # 성능 측정 반복 횟수 (최소 10회)
PERF_SIZES = [3, 5, 13, 25]
MODE1_SIZE = 3          # 모드 1은 3x3 고정

LABEL_CROSS = "Cross"   # 표준 라벨
LABEL_X = "X"           # 표준 라벨
LABEL_UNDECIDED = "UNDECIDED"

# 라벨 정규화 표 (소문자 기준)
#   expected 값: '+' -> Cross, 'x' -> X
#   filter 키  : 'cross' -> Cross, 'x' -> X
LABEL_MAP = {
    "+": LABEL_CROSS,
    "cross": LABEL_CROSS,
    "x": LABEL_X,
}

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
LINE = "#" + "-" * 40


# ----------------------------------------------------------------------
# 1) 데이터 구조: n x n 행렬 (패턴/필터 공용)
# ----------------------------------------------------------------------
class Matrix:
    """n x n 2차원 배열을 저장하는 단순 자료구조."""

    def __init__(self, size, data=None):
        self.size = size
        if data is None:
            self.data = [[0.0] * size for _ in range(size)]
        else:
            self.data = data

    def set(self, row, col, value):
        """(row, col) 위치에 값을 저장한다."""
        self.data[row][col] = float(value)

    def get(self, row, col):
        """(row, col) 위치의 값을 읽어온다."""
        return self.data[row][col]

    def flatten(self):
        """보너스: 2차원 배열을 길이 N*N 의 1차원 배열로 변환한다 (행 우선)."""
        flat = []
        for row in self.data:
            for value in row:
                flat.append(value)
        return flat

    def show(self, indent="  "):
        for row in self.data:
            print(indent + " ".join(fmt_number(v) for v in row))

    @staticmethod
    def from_rows(rows):
        """JSON 등에서 읽은 2차원 리스트를 검증하고 Matrix로 만든다.

        검증 실패 시 ValueError(사유)를 발생시킨다.
        """
        if not isinstance(rows, list) or len(rows) == 0:
            raise ValueError("2차원 배열이 아님(비어 있거나 list가 아님)")
        n = len(rows)
        data = []
        for r, row in enumerate(rows):
            if not isinstance(row, list):
                raise ValueError("{}번째 행이 list가 아님".format(r + 1))
            if len(row) != n:
                raise ValueError(
                    "정사각형이 아님: 행 {}개, {}번째 행의 열 {}개".format(n, r + 1, len(row))
                )
            new_row = []
            for c, value in enumerate(row):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("({}, {}) 값이 숫자가 아님: {!r}".format(r + 1, c + 1, value))
                if not math.isfinite(value):
                    raise ValueError("({}, {}) 값이 유한한 숫자가 아님".format(r + 1, c + 1))
                new_row.append(float(value))
            data.append(new_row)
        return Matrix(n, data)


# ----------------------------------------------------------------------
# 출력 보조 함수
# ----------------------------------------------------------------------
def fmt_number(value):
    """행렬 값 출력용 (1.0 -> 1, 0.9 -> 0.9)."""
    return "{:g}".format(value)


def fmt_score(value):
    """점수 출력용: 부동소수점 오차(0.8999999999999999 등)가 보이도록 repr 형식으로 표시."""
    return repr(float(value))


def print_header(title):
    print()
    print(LINE)
    print("# " + title)
    print(LINE)


# ----------------------------------------------------------------------
# 2) 라벨 정규화
# ----------------------------------------------------------------------
def normalize_label(raw):
    """expected 값 / filter 키를 표준 라벨(Cross, X)로 정규화한다.

    알 수 없는 값이면 ValueError 를 발생시킨다.
    """
    if not isinstance(raw, str):
        raise ValueError("라벨이 문자열이 아님: {!r}".format(raw))
    key = raw.strip().lower()
    if key not in LABEL_MAP:
        raise ValueError("알 수 없는 라벨: {!r}".format(raw))
    return LABEL_MAP[key]


# ----------------------------------------------------------------------
# 3) MAC 연산 (반복문 직접 구현, 외부 라이브러리 금지)
# ----------------------------------------------------------------------
def mac(pattern, flt):
    """패턴과 필터를 위치별로 곱하고 모두 더한다. (2차원 접근)"""
    if pattern.size != flt.size:
        raise ValueError("크기 불일치: 패턴 {0}x{0} vs 필터 {1}x{1}".format(pattern.size, flt.size))
    n = pattern.size
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += pattern.data[i][j] * flt.data[i][j]
    return total


def mac_flat(pattern_flat, filter_flat):
    """보너스: 1차원 배열(길이 N*N) 버전 MAC 연산."""
    if len(pattern_flat) != len(filter_flat):
        raise ValueError("크기 불일치: 길이 {} vs {}".format(len(pattern_flat), len(filter_flat)))
    total = 0.0
    for k in range(len(pattern_flat)):
        total += pattern_flat[k] * filter_flat[k]
    return total


# ----------------------------------------------------------------------
# 4) 점수 비교 정책 (epsilon)
# ----------------------------------------------------------------------
def compare_scores(score_a, score_b):
    """'A' / 'B' / 'TIE' 반환. |a - b| < EPSILON 이면 동점."""
    if abs(score_a - score_b) < EPSILON:
        return "TIE"
    return "A" if score_a > score_b else "B"


# ----------------------------------------------------------------------
# 5) 성능 측정 (I/O 제외, 연산 함수 호출 구간만 측정)
# ----------------------------------------------------------------------
def measure_ms(func, *args, repeat=REPEAT):
    """func(*args)를 repeat회 호출한 평균 시간(ms)을 반환한다."""
    start = time.perf_counter()
    for _ in range(repeat):
        func(*args)
    elapsed = time.perf_counter() - start
    return elapsed / repeat * 1000.0


def print_perf_table(rows):
    """rows: [(size, avg_ms), ...]"""
    print("크기        평균 시간(ms)    연산 횟수(N²)")
    print("-" * 40)
    for size, avg_ms in rows:
        label = "{0}×{0}".format(size)
        print("{:<10}{:>14.4f}{:>14}".format(label, avg_ms, size * size))


# ----------------------------------------------------------------------
# 6) 패턴 생성기 (보너스)
# ----------------------------------------------------------------------
def generate_cross(n):
    """N x N 십자가(Cross) 패턴 생성: 가운데 행 + 가운데 열이 1."""
    mid = n // 2
    m = Matrix(n)
    for i in range(n):
        for j in range(n):
            if i == mid or j == mid:
                m.set(i, j, 1.0)
    return m


def generate_x(n):
    """N x N X 패턴 생성: 두 대각선이 1."""
    m = Matrix(n)
    for i in range(n):
        for j in range(n):
            if i == j or i + j == n - 1:
                m.set(i, j, 1.0)
    return m


# ----------------------------------------------------------------------
# 7) 모드 1: 사용자 입력 (3x3)
# ----------------------------------------------------------------------
def read_matrix_from_console(title, n=MODE1_SIZE):
    """콘솔에서 n줄(공백 구분)을 입력받는다. 오류 시 안내 후 재입력."""
    while True:
        print("{} ({}줄 입력, 공백 구분)".format(title, n))
        lines = [input() for _ in range(n)]

        errors = []
        rows = []
        for idx, line in enumerate(lines, start=1):
            tokens = line.split()
            if len(tokens) != n:
                errors.append("{}번째 줄: 숫자 {}개 입력됨 (필요: {}개)".format(idx, len(tokens), n))
                continue
            try:
                row = [float(t) for t in tokens]
            except ValueError:
                errors.append("{}번째 줄: 숫자로 변환할 수 없는 값이 있음".format(idx))
                continue
            if not all(math.isfinite(v) for v in row):
                errors.append("{}번째 줄: nan/inf 는 사용할 수 없음".format(idx))
                continue
            rows.append(row)

        if errors:
            print("입력 형식 오류: 각 줄에 {}개의 숫자를 공백으로 구분해 입력하세요.".format(n))
            for message in errors:
                print("  - " + message)
            print("→ {} 입력을 처음부터 다시 진행합니다.\n".format(title))
            continue
        return Matrix(n, rows)


def show_ab_result(pattern, filter_a, filter_b):
    """A/B 필터와 패턴의 MAC 결과 + 판정을 출력하고 평균 연산 시간(ms)을 반환한다.

    모드 1과 보너스(패턴 생성기)가 함께 사용한다.
    """
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    avg_ms = (measure_ms(mac, pattern, filter_a) + measure_ms(mac, pattern, filter_b)) / 2

    result = compare_scores(score_a, score_b)
    print("A 점수: {}".format(fmt_score(score_a)))
    print("B 점수: {}".format(fmt_score(score_b)))
    print("연산 시간(평균/{}회): {:.4f} ms".format(REPEAT, avg_ms))
    if result == "TIE":
        print("판정: 판정 불가 (|A-B| < {})".format(EPSILON))
    else:
        print("판정: {}".format(result))
    return avg_ms


def run_mode1():
    print_header("[1] 필터 입력")
    filter_a = read_matrix_from_console("필터 A")
    print()
    filter_b = read_matrix_from_console("필터 B")

    print("\n✓ 필터 A, B 저장 완료")
    print("필터 A:")
    filter_a.show()
    print("필터 B:")
    filter_b.show()

    print_header("[2] 패턴 입력")
    pattern = read_matrix_from_console("패턴")
    print("\n✓ 패턴 저장 완료")

    print_header("[3] MAC 결과")
    avg_ms = show_ab_result(pattern, filter_a, filter_b)

    print_header("[4] 성능 분석 (평균/{}회)".format(REPEAT))
    print_perf_table([(MODE1_SIZE, avg_ms)])


# ----------------------------------------------------------------------
# 8) 모드 2: data.json 분석
# ----------------------------------------------------------------------
def load_json_file(path):
    """data.json 로드. 실패 시 (None, 사유) 반환 (프로그램은 중단하지 않는다)."""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, "파일을 찾을 수 없음: {}".format(path)
    except json.JSONDecodeError as e:
        return None, "JSON 형식 오류: {}".format(e)
    except OSError as e:
        return None, "파일 읽기 오류: {}".format(e)


def load_filters(data):
    """filters 를 읽어 {N: {"Cross": Matrix, "X": Matrix}} 또는 {N: "오류 사유"} 로 반환한다."""
    filters = {}
    raw_filters = data.get("filters")
    if not isinstance(raw_filters, dict):
        print("✗ filters 항목이 없거나 형식이 올바르지 않습니다.")
        return filters

    for key, entry in raw_filters.items():
        match = re.fullmatch(r"size_(\d+)", key)
        if not match:
            print("✗ {} 필터 키 형식 오류 (size_N 형태가 아님)".format(key))
            continue
        n = int(match.group(1))
        try:
            if not isinstance(entry, dict):
                raise ValueError("필터 묶음이 dict가 아님")
            pair = {}
            for raw_label, rows in entry.items():
                label = normalize_label(raw_label)          # 필터 키 정규화
                matrix = Matrix.from_rows(rows)
                if matrix.size != n:
                    raise ValueError(
                        "{} 필터 크기 불일치: 키는 {}인데 실제 {}x{}".format(
                            raw_label, key, matrix.size, matrix.size)
                    )
                pair[label] = matrix
            if LABEL_CROSS not in pair or LABEL_X not in pair:
                raise ValueError("Cross/X 필터가 모두 필요함 (현재: {})".format(", ".join(pair) or "없음"))
            filters[n] = pair
            print("✓ {:<8}필터 로드 완료 ({}, {})".format(key, LABEL_CROSS, LABEL_X))
        except ValueError as e:
            filters[n] = str(e)
            print("✗ {:<8}필터 로드 실패: {}".format(key, e))
    return filters


def analyze_case(key, entry, filters, samples):
    """패턴 1건을 분석해 결과 dict 를 반환한다. 어떤 오류도 케이스 단위 FAIL 로 처리한다."""
    result = {"key": key, "status": "FAIL", "reason": "", "cross": None, "x": None,
              "decision": None, "expected": None}
    try:
        # (1) 키에서 N 추출
        match = re.fullmatch(r"size_(\d+)_(\d+)", key)
        if not match:
            raise ValueError("패턴 키 형식 오류 (size_N_idx 형태가 아님)")
        n = int(match.group(1))

        # (2) 스키마 확인
        if not isinstance(entry, dict) or "input" not in entry or "expected" not in entry:
            raise ValueError("스키마 오류: input/expected 항목이 필요함")

        # (3) expected 라벨 정규화
        expected = normalize_label(entry["expected"])
        result["expected"] = expected

        # (4) 해당 size_N 필터 선택
        if n not in filters:
            raise ValueError("size_{} 필터가 data.json 에 없음".format(n))
        pair = filters[n]
        if isinstance(pair, str):
            raise ValueError("size_{} 필터 사용 불가 ({})".format(n, pair))

        # (5) 패턴 검증 + 필터/패턴 크기 일치 검증
        pattern = Matrix.from_rows(entry["input"])
        flt_size = pair[LABEL_CROSS].size
        if pattern.size != flt_size:
            raise ValueError("크기 불일치: 패턴 {0}x{0} vs 필터 {1}x{1}".format(pattern.size, flt_size))

        # (6) MAC 연산 + 판정
        score_cross = mac(pattern, pair[LABEL_CROSS])
        score_x = mac(pattern, pair[LABEL_X])
        result["cross"] = score_cross
        result["x"] = score_x
        samples.setdefault(n, pattern)

        cmp_result = compare_scores(score_cross, score_x)
        if cmp_result == "TIE":
            decision = LABEL_UNDECIDED
        else:
            decision = LABEL_CROSS if cmp_result == "A" else LABEL_X
        result["decision"] = decision

        # (7) expected 비교 (표준 라벨 기준)
        if decision == LABEL_UNDECIDED:
            result["reason"] = "동점(UNDECIDED) 처리 규칙에 따라 FAIL"
        elif decision != expected:
            result["reason"] = "판정({})이 expected({})와 다름".format(decision, expected)
        else:
            result["status"] = "PASS"
    except ValueError as e:
        result["reason"] = str(e)
    except Exception as e:  # 예기치 못한 오류도 프로그램을 중단시키지 않는다
        result["reason"] = "예기치 못한 오류: {}: {}".format(type(e).__name__, e)
    return result


def run_mode2():
    data, error = load_json_file(DATA_FILE)
    if error:
        print("\n[오류] " + error)
        print("data.json 을 main.py 와 같은 폴더에 두고 다시 실행하세요.")
        return
    if not isinstance(data, dict):
        print("\n[오류] data.json 최상위 구조가 객체(dict)가 아닙니다.")
        return

    print_header("[1] 필터 로드")
    filters = load_filters(data)

    print_header("[2] 패턴 분석 (라벨 정규화 적용)")
    patterns = data.get("patterns")
    if not isinstance(patterns, dict):
        print("✗ patterns 항목이 없거나 형식이 올바르지 않습니다. (총 0건)")
        patterns = {}

    samples = {}      # 크기별 첫 번째 유효 패턴 (성능 측정용)
    results = []
    for key, entry in patterns.items():
        r = analyze_case(key, entry, filters, samples)
        results.append(r)
        print("--- {} ---".format(key))
        if r["cross"] is not None:
            print("Cross 점수: {}".format(fmt_score(r["cross"])))
            print("X 점수: {}".format(fmt_score(r["x"])))
            line = "판정: {} | expected: {} | {}".format(r["decision"], r["expected"], r["status"])
            if r["status"] == "FAIL":
                line += " ({})".format(r["reason"])
            print(line)
        else:
            print("판정: 불가 | expected: {} | FAIL ({})".format(r["expected"] or "-", r["reason"]))

    print_header("[3] 성능 분석 (평균/{}회)".format(REPEAT))
    perf_rows = []
    for n in PERF_SIZES:
        if n == 3:
            # data.json 에 3x3 필터가 없으므로 패턴 생성기(보너스)의 3x3 패턴을 재활용
            flt = generate_cross(3)
            pat = generate_cross(3)
        else:
            pair = filters.get(n)
            if pair is None or isinstance(pair, str):
                print("(size_{} 필터를 사용할 수 없어 성능 측정을 건너뜀)".format(n))
                continue
            flt = pair[LABEL_CROSS]
            pat = samples.get(n, flt)
        perf_rows.append((n, measure_ms(mac, pat, flt)))
    print_perf_table(perf_rows)

    print_header("[4] 결과 요약")
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = total - passed
    print("총 테스트: {}개".format(total))
    print("통과: {}개".format(passed))
    print("실패: {}개".format(failed))
    if failed:
        print("\n실패 케이스:")
        for r in results:
            if r["status"] == "FAIL":
                print("- {}: {}".format(r["key"], r["reason"]))
    print('\n(상세 원인 분석 및 복잡도 설명은 README.md 의 "결과 리포트" 섹션 참고)')


# ----------------------------------------------------------------------
# 9) 모드 3: 보너스 (패턴 생성기 + 1차원 배열 최적화 비교)
# ----------------------------------------------------------------------
def read_size_from_console():
    while True:
        text = input("생성할 패턴 크기 N 입력 (3 이상 정수, 예: 5): ").strip()
        try:
            n = int(text)
        except ValueError:
            print("입력 형식 오류: 정수를 입력하세요.")
            continue
        if n < 3:
            print("입력 형식 오류: 3 이상의 정수를 입력하세요.")
            continue
        return n


def run_bonus():
    print_header("[B-1] 패턴 생성기")
    n = read_size_from_console()
    cross = generate_cross(n)
    x_pat = generate_x(n)
    print("\n생성된 Cross 패턴 ({0}x{0}):".format(n))
    cross.show()
    print("생성된 X 패턴 ({0}x{0}):".format(n))
    x_pat.show()

    print_header("[B-2] 생성 패턴으로 판정 (모드 1 판정 로직 재사용, A=Cross, B=X)")
    print("[패턴 = 생성된 Cross]")
    show_ab_result(cross, cross, x_pat)
    print("\n[패턴 = 생성된 X]")
    show_ab_result(x_pat, cross, x_pat)

    print_header("[B-3] 2차원 vs 1차원 배열 성능 비교 (평균/{}회)".format(REPEAT))
    sizes = list(PERF_SIZES)
    if n not in sizes:
        sizes.append(n)
    print("크기        연산 횟수    2차원(ms)    1차원(ms)    개선율    결과 일치")
    print("-" * 66)
    for size in sizes:
        pat = generate_cross(size)
        flt = generate_x(size)
        pat_flat = pat.flatten()          # 변환 비용은 측정 구간에서 제외
        flt_flat = flt.flatten()
        t2d = measure_ms(mac, pat, flt)
        t1d = measure_ms(mac_flat, pat_flat, flt_flat)
        same = abs(mac(pat, flt) - mac_flat(pat_flat, flt_flat)) < EPSILON
        gain = (t2d - t1d) / t2d * 100.0 if t2d > 0 else 0.0
        print("{:<10}{:>10}{:>13.4f}{:>13.4f}{:>9.1f}%{:>9}".format(
            "{0}×{0}".format(size), size * size, t2d, t1d, gain, "OK" if same else "NG"))


# ----------------------------------------------------------------------
# 10) 메인
# ----------------------------------------------------------------------
def select_mode():
    print("=== Mini NPU Simulator ===\n")
    print("[모드 선택]\n")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    print("3. (보너스) 패턴 생성기 + 1차원 배열 성능 비교")
    while True:
        choice = input("선택: ").strip()
        if choice in ("1", "2", "3"):
            return choice
        print("입력 형식 오류: 1, 2, 3 중 하나를 입력하세요.")


def main():
    try:
        choice = select_mode()
        if choice == "1":
            run_mode1()
        elif choice == "2":
            run_mode2()
        else:
            run_bonus()
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 중단되어 프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
