# CLAUDE.md

## 주가 데이터 NaN 보간 규칙 (전처리)

블룸버그 등에서 추출한 원본 OHLCV 데이터의 NaN을 채우는 규칙.
규칙은 순서대로 적용된다.

### 규칙 1: OHLC 전체 NaN → forward-fill Close, O/H/L = Close, Volume = 0

당일 거래가 아예 없어서 가격 데이터가 모두 비어있는 경우.
- Close를 `__Close_ffilled_temp__` (사전에 ffill한 임시 컬럼)로 채움
- Open, High, Low를 채워진 Close 값으로 설정
- Volume은 무조건 0 (거래가 없었으므로)
- 일부 OHLC 컬럼만 존재하는 경우에도 존재하는 컬럼 기준으로 동일 로직 적용

### 규칙 2: Close는 있는데 O/H/L이 NaN → O/H/L = Close

규칙 1과 별개로, 부분적으로 누락된 경우를 처리.
Close가 유효한데 Open/High/Low 중 일부가 NaN이면 Close 값으로 채움.

### 규칙 3: O == H == L == C이고 Volume이 NaN → Volume = 0

OHLC가 모두 동일한 유효값 = 가격 변동 없음 = 거래가 없었을 가능성.
Volume만 누락된 경우 0으로 채움.
OHLC + Volume 5개 컬럼이 모두 존재할 때만 적용.

## Feature (value, flag) 인코딩

상장 이전, MA warmup 기간 등 유효하지 않은 피처 값을 처리하는 방식.
행을 삭제하지 않고, 각 피처를 2차원 벡터로 확장한다.

- 유효한 값: `(value, 0)`
- 유효하지 않은 값 (NaN): `(0, 1)`

flag 0 = 유효, flag 1 = 무효.

### 적용 대상
- 상장 이전 (pre-listing): 가격 데이터 자체가 없음
- Rolling warmup: MA20, RSI14 등 지표 계산에 필요한 최소 기간 미충족
- 기타 NaN: 어떤 이유로든 피처 값이 비어있는 경우

### 적용하지 않는 대상
- tradable_mask: 기존대로 별도 binary 벡터
- portfolio weights, cumulative PnL: 항상 유효 (env에서 계산)
