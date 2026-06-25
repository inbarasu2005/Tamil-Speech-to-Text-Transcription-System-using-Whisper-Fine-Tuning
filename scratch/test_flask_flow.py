import urllib.request
import urllib.parse
import urllib.error
import http.cookiejar
import json
import sys

# Configure cookies to handle sessions across requests
cookie_jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
urllib.request.install_opener(opener)

import time
BASE_URL = "http://127.0.0.1:5000"
TEST_EMAIL = f"tester_{int(time.time())}@example.com"

def test_login_required():
    print("[TEST] Verifying login required on Dashboard...")
    try:
        # Fetch dashboard which should redirect to /login
        response = urllib.request.urlopen(BASE_URL + "/")
        final_url = response.geturl()
        print(f"   Dashboard access URL redirected to: {final_url}")
        
        # Check if we were redirected to login
        if "/login" in final_url:
            print("   [PASS] Successfully blocked and redirected to login page.")
            return True
        else:
            print("   [FAIL] Dashboard did not redirect unauthorized user to login.")
            return False
    except Exception as e:
        print(f"   [ERROR] Failed to verify: {e}")
        return False

def test_registration():
    print("[TEST] Registering new user...")
    register_url = BASE_URL + "/register"
    data = urllib.parse.urlencode({
        'fullname': 'Integration Tester',
        'email': TEST_EMAIL,
        'password': 'password123',
        'confirm_password': 'password123'
    }).encode('utf-8')
    
    try:
        # POST register data
        req = urllib.request.Request(register_url, data=data, method="POST")
        response = urllib.request.urlopen(req)
        final_url = response.geturl()
        print(f"   Registered user redirect: {final_url}")
        
        # Check if redirected to login page after successful registration
        if "/login" in final_url:
            print("   [PASS] User successfully registered and redirected.")
            return True
        else:
            print("   [FAIL] Registration did not redirect to login page.")
            return False
    except Exception as e:
        print(f"   [ERROR] Failed to register: {e}")
        return False

def test_login():
    print("[TEST] Logging in user...")
    login_url = BASE_URL + "/login"
    data = urllib.parse.urlencode({
        'email': TEST_EMAIL,
        'password': 'password123',
        'remember': 'true'
    }).encode('utf-8')
    
    try:
        req = urllib.request.Request(login_url, data=data, method="POST")
        response = urllib.request.urlopen(req)
        final_url = response.geturl()
        print(f"   Login redirect: {final_url}")
        
        # Verify redirect to dashboard
        if final_url.rstrip('/') == BASE_URL:
            print("   [PASS] Login successful, redirected to Dashboard.")
            return True
        else:
            print("   [FAIL] Login failed, redirection incorrect.")
            return False
    except Exception as e:
        print(f"   [ERROR] Login exception: {e}")
        return False

def test_dashboard_access_after_login():
    print("[TEST] Checking dashboard access after login...")
    try:
        # Re-fetch dashboard with session cookies present
        response = urllib.request.urlopen(BASE_URL + "/")
        final_url = response.geturl()
        print(f"   Dashboard URL: {final_url}")
        
        if final_url.rstrip('/') == BASE_URL:
            # Verify page content has the text Welcome, Integration Tester
            html_content = response.read().decode('utf-8')
            if "Integration Tester" in html_content:
                print("   [PASS] Successfully retrieved Dashboard with correct session details.")
                return True
            else:
                print("   [FAIL] Dashboard was loaded but session details were missing or incorrect.")
                return False
        else:
            print("   [FAIL] Dashboard request was redirected unexpectedly.")
            return False
    except Exception as e:
        print(f"   [ERROR] Dashboard access error: {e}")
        return False

def test_logout():
    print("[TEST] Testing logout function...")
    logout_url = BASE_URL + "/logout"
    try:
        response = urllib.request.urlopen(logout_url)
        final_url = response.geturl()
        print(f"   Logout redirect: {final_url}")
        
        if "/login" in final_url:
            # Let's verify dashboard is blocked again
            try:
                check_resp = urllib.request.urlopen(BASE_URL + "/")
                check_url = check_resp.geturl()
                if "/login" in check_url:
                    print("   [PASS] Logout succeeded. Session destroyed and dashboard access blocked.")
                    return True
                else:
                    print("   [FAIL] Dashboard still accessible after logging out.")
                    return False
            except Exception as e:
                print(f"   [ERROR] Access verification error: {e}")
                return False
        else:
            print("   [FAIL] Logout did not redirect to login page.")
            return False
    except Exception as e:
        print(f"   [ERROR] Logout exception: {e}")
        return False

if __name__ == "__main__":
    tests = [
        test_login_required,
        test_registration,
        test_login,
        test_dashboard_access_after_login,
        test_logout
    ]
    
    passed_all = True
    for t in tests:
        if not t():
            passed_all = False
            break
            
    if passed_all:
        print("\n====================================")
        print("ALL AUTHENTICATION FLOW TESTS PASSED")
        print("====================================")
        sys.exit(0)
    else:
        print("\n====================================")
        print("SOME TESTS FAILED")
        print("====================================")
        sys.exit(1)
