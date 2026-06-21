import requests, time

API = "http://127.0.0.1:8000"

def test(n, name, fn):
    try:
        result = fn()
        print(f"[TEST {n}] {name}: PASS -> {result}")
        return True
    except Exception as e:
        print(f"[TEST {n}] {name}: FAIL -> {e}")
        return False

def t1():
    r = requests.get(API + "/", timeout=5)
    assert r.status_code == 200
    return r.json()["message"]

def t2():
    r = requests.post(API + "/query", json={"question": "test"}, timeout=5)
    return "status=" + str(r.status_code)

def t3():
    path = "D:/desktop/git_test/rag-qa-system/docs/sample_docs/sample.txt"
    with open(path, "rb") as f:
        r = requests.post(API + "/upload", files={"file": f}, timeout=15)
    data = r.json()
    assert data["status"] == "ok"
    return f"status=ok, chunks={data['chunks']}"

def t4():
    r = requests.post(API + "/query", json={"question": "RAG 的核心流程是什么？"}, timeout=15)
    return r.json()["answer"][:100] + "..."

def t5():
    r = requests.post(API + "/clear", timeout=5)
    return r.json()["message"]

all_pass = True
all_pass &= test(1, "GET /", t1)
all_pass &= test(2, "Query without KB", t2)

# Server needs restart after clear from previous run
# Let's upload first then query
all_pass &= test(3, "Upload document", t3)

if all_pass:
    all_pass &= test(4, "Query after upload", t4)

all_pass &= test(5, "Clear knowledge base", t5)

print()
if all_pass:
    print("=== ALL TESTS PASSED ===")
else:
    print("=== SOME TESTS FAILED ===")
    exit(1)
