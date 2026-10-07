import time
import random

random.seed(42)

def test_l3():
    t0 = time.time()
    samples = []
    for i in range(500):
        cat = i % 8
        if cat == 0:
            b = random.randint(3, 30)
            e = random.randint(100, 2000)
            m = random.choice([17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 79, 83, 89, 97])
            code = f"print(pow({b}, {e}, {m}))"
        elif cat == 1:
            n = random.randint(10000, 999999)
            code = f"def f(n):\n d, res = 2, []\n while d*d <= n:\n  while n%d == 0:\n   res.append(d)\n   n //= d\n  d += 1\n if n > 1: res.append(n)\n return res\nprint(f({n}))"
        elif cat == 2:
            data = [random.randint(5, 500) for _ in range(random.randint(6, 12))]
            code = f"import statistics\nd = {data}\nprint(statistics.mean(d), statistics.median(d))"
        elif cat == 3:
            n = random.randint(15, 40)
            k = random.randint(3, 8)
            code = f"import math\nprint(math.comb({n}, {k}))"
        elif cat == 4:
            a, b = random.randint(1000, 50000), random.randint(1000, 50000)
            code = f"import math\ng = math.gcd({a}, {b})\nprint(g, ({a}*{b})//g)"
        elif cat == 5:
            s = random.randint(50, 2000)
            code = f"def c(n):\n s = 0\n while n != 1:\n  n = n//2 if n%2==0 else 3*n+1\n  s += 1\n return s\nprint(c({s}))"
        elif cat == 6:
            m = [[random.randint(-5, 5) for _ in range(3)] for _ in range(3)]
            code = f"A = {m}\ndet = A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1]) - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]) + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0])\nprint(det)"
        else:
            w = random.choice(["tensor", "kernel", "gradient", "transformer", "recurrent", "attention"])
            salt = random.randint(1000, 9999)
            code = f"import hashlib\nprint(hashlib.sha256('{w}_{salt}'.encode()).hexdigest())"
        
        # Test exec
        loc = {}
        exec(code, {}, loc)
        samples.append(code)

    elapsed = time.time() - t0
    print(f"Generated and executed {len(samples)} python math codes in {elapsed:.2f} seconds!")

test_l3()
