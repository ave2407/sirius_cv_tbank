import os, re, requests, zipfile
from urllib.parse import quote

PASSWORD = r"7*5\Lq=Oik"
BASE = "https://data.tinkoff.ru"
TOKEN = "YsqPKQkapc5xKMb"
SHARE_URL = f"{BASE}/s/{TOKEN}"
OUT_DIR = "data"
OUT_NAME = "data_for_sirius_2025.zip"
SAVE_TO  = os.path.join(OUT_DIR, OUT_NAME)
UNZIP_TO = os.path.join(OUT_DIR, "raw")

os.makedirs(OUT_DIR, exist_ok=True)
sess = requests.Session()
sess.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Referer": SHARE_URL,
})

def extract_tokens(html: str):
    rt = re.search(r'name="requesttoken"\s+value="([^"]+)"', html)
    st = re.search(r'name="sharingToken"\s+value="([^"]+)"', html)
    if not rt:
        rt = re.search(r'<head[^>]*data-requesttoken="([^"]+)"', html)
    requesttoken = rt.group(1) if rt else None
    sharingToken = st.group(1) if st else TOKEN  # запасной вариант
    return requesttoken, sharingToken

r = sess.get(SHARE_URL, allow_redirects=True)
r.raise_for_status()
requesttoken, sharingToken = extract_tokens(r.text)
if not requesttoken:
    raise RuntimeError("Не нашли requesttoken на странице (HTML структуры могли измениться).")

auth_attempts = [
    (f"{SHARE_URL}/authenticate/downloadShare", {"password": PASSWORD, "sharingToken": sharingToken}),
    (f"{BASE}/index.php/apps/files_sharing/publicpreview", {"password": PASSWORD, "shareToken": TOKEN}),
    (f"{BASE}/index.php/apps/files_sharing/ajax/sharepassword.php", {"password": PASSWORD, "sharingToken": sharingToken}),
    (SHARE_URL, {"password": PASSWORD, "requesttoken": requesttoken, "sharingToken": sharingToken}),
]

authed = False
for url, data in auth_attempts:
    try:
        resp = sess.post(url, data=data, headers={"requesttoken": requesttoken, "Referer": SHARE_URL}, allow_redirects=True)
        # 200 или 204 обычно ок; некоторые отвечают JSON/пусто.
        if resp.status_code in (200, 204):
            authed = True
            break
    except requests.RequestException:
        pass

if not authed:
    raise RuntimeError("Авторизация не прошла ни на одном эндпоинте. Проверь пароль/доступ.")

dl_candidates = [
    f"{SHARE_URL}/download/{OUT_NAME}",
    f"{SHARE_URL}/download?path=&files={quote(OUT_NAME)}",
    f"{BASE}/index.php/s/{TOKEN}/download/{OUT_NAME}",
    f"{BASE}/index.php/s/{TOKEN}/download?path=&files={quote(OUT_NAME)}",
]

got = False
last_content = None
for url in dl_candidates:
    try:
        with sess.get(url, stream=True, allow_redirects=True, headers={"Referer": SHARE_URL}) as dl:
            ct = (dl.headers.get("Content-Type") or "").lower()
            cd = dl.headers.get("Content-Disposition") or ""
            if dl.status_code == 200 and ("zip" in ct or "attachment" in cd or "octet-stream" in ct):
                with open(SAVE_TO, "wb") as f:
                    for chunk in dl.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
                got = True
                break
            else:
                last_content = dl.content  # для диагностики
    except requests.RequestException:
        continue

if not got:
    dbg = os.path.join(OUT_DIR, "debug_download_response.html")
    with open(dbg, "wb") as f:
        f.write(last_content or b"(empty)")
    raise RuntimeError(f"Сервер вернул не ZIP. Сохранил ответ сюда: {dbg}")

try:
    with zipfile.ZipFile(SAVE_TO) as zf:
        os.makedirs(UNZIP_TO, exist_ok=True)
        zf.extractall(UNZIP_TO)
    print("OK: файл скачан и распакован.")
    print("ZIP:", SAVE_TO)
    print("Папка:", UNZIP_TO)
except zipfile.BadZipFile:
    print("Файл скачан, но это не ZIP. Оставил как есть:", SAVE_TO)
