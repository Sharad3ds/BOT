import requests
import pyotp
import hashlib
from urllib import parse

CLIENT_ID    = "2NWM6G5K7Y-200"
SECRET_KEY   = "UIN2uCF0UYogoTOU"
REDIRECT_URI = "https://fyers.in"
FY_ID        = "FAD11910"
PIN          = "8866"
TOTP_KEY     = "PKQ36PCDIDG7D6DBWPD723WG2HRTRGGR"  # ← New key
APP_ID_TYPE  = "2"

APP_ID_HASH = hashlib.sha256(
    f"{CLIENT_ID}:{SECRET_KEY}".encode()
).hexdigest()

BASE_URL   = "https://api-t2.fyers.in/vagator/v2"
BASE_URL_2 = "https://api-t1.fyers.in/api/v3"

# Step 1
r1 = requests.post(f"{BASE_URL}/send_login_otp",
    json={"fy_id": FY_ID, "app_id": APP_ID_TYPE})
request_key = r1.json()["request_key"]
print("✅ Step 1 - OTP sent")

# Step 2
totp = pyotp.TOTP(TOTP_KEY).now()
r2 = requests.post(f"{BASE_URL}/verify_otp",
    json={"request_key": request_key, "otp": totp})
print(f"Step 2 response: {r2.json()}")
request_key = r2.json()["request_key"]
print("✅ Step 2 - TOTP verified")

# Step 3
r3 = requests.post(f"{BASE_URL}/verify_pin",
    json={"request_key": request_key, "identity_type": "pin", "identifier": PIN})
access_token = r3.json()["data"]["access_token"]
print("✅ Step 3 - PIN verified")

# Step 4
r4 = requests.post(f"{BASE_URL_2}/token",
    json={
        "fyers_id"      : FY_ID,
        "app_id"        : "2NWM6G5K7Y",
        "redirect_uri"  : REDIRECT_URI,
        "appType"       : "200",
        "code_challenge": "",
        "state"         : "sample_state",
        "scope"         : "",
        "nonce"         : "",
        "response_type" : "code",
        "create_cookie" : True
    },
    headers={"Authorization": f"Bearer {access_token}"}
)
print(f"✅ Step 4 - Status: {r4.status_code}")
url = r4.json()["Url"]
auth_code = parse.parse_qs(parse.urlparse(url).query)["auth_code"][0]
print("✅ Auth code obtained")

# Step 5
r5 = requests.post(f"{BASE_URL_2}/validate-authcode",
    json={
        "grant_type": "authorization_code",
        "appIdHash" : APP_ID_HASH,
        "code"      : auth_code,
    }
)
print(f"\n🎉 Final Response: {r5.json()}")

if r5.json().get("access_token"):
    token = r5.json()["access_token"]
    with open("fyers_token.txt", "w") as f:
        f.write(token)
    with open("fyers_appid.txt", "w") as f:
        f.write(CLIENT_ID)
    print(f"\n✅ Token saved successfully!")

