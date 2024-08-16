import torch
import time

omega = 30
scale = 10

def finder(x):
     torch.sin(omega * (torch.abs(x) + 1) * x)

def benchmark_activation(activation_func, input_tensor, repeats=100):
    """활성화 함수에 대한 벤치마크를 수행하고 평균 실행 시간을 반환합니다."""
    times = []
    
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        _ = activation_func(input_tensor)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return sum(times) / repeats

def run_benchmark(input_size, input_range, repeats=100):
    """주어진 입력 크기와 범위에 대해 벤치마크를 실행."""
    # 입력 텐서 생성 (주어진 범위에서 랜덤 값 생성)
    input_tensor = torch.rand(input_size, device=device) * (input_range[1] - input_range[0]) + input_range[0]

    # 활성화 함수 리스트
    activation_functions = {
        'finer' : lambda x : torch.sin(omega * (torch.abs(x) + 1) * x),
        'ReLU': torch.relu,
        'Sin': torch.sin,
        'Exp': torch.exp,
        'Gauss': lambda x: torch.exp(-(x*omega)**2),
        'wireReal': lambda x: torch.cos(x*omega)*torch.exp(-(x*scale)**2),
        'wireComplex': lambda x: torch.exp(1j*x*omega - (x*scale).abs().square()),
    }

    # 각 활성화 함수에 대해 벤치마크 수행 및 결과 출력
    for name, func in activation_functions.items():
        avg_time = benchmark_activation(func, input_tensor, repeats)
        print(f"{name} (범위: {input_range}) 평균 실행 시간: {avg_time:.3f} ms")


# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 테스트할 텐서 크기
input_size = (512,512,256,4) 
print(f"테스트할 텐서의 크기 : {input_size}")

# 반복 횟수 (평균 시간을 계산하기 위해 여러 번 실행)
repeats = 1000
print(f"반복 횟수 (결과는 평균): {repeats}")

# 테스트할 입력 데이터 범위 리스트
input_ranges = [(-1, 1), (-10, 10), (-100, 100), (-1000, 1000)]

# 각 범위에 대해 벤치마크 수행
for input_range in input_ranges:
    print(f"입력 데이터 범위: {input_range}")
    run_benchmark(input_size, input_range, repeats)
    print("\n")
