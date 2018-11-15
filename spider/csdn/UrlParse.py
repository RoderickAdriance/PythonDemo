import requests
import json
headers={
    "Cookie":"uuid_tt_dd=10_30312377400-1525341458653-196857; ADHOC_MEMBERSHIP_CLIENT_ID1.0=3725ff2d-f0d2-e21d-7c43-d100e54b4a00; UM_distinctid=16328b1d18d2e0-03c7beb9f835-3c3c5905-15f900-16328b1d18ea36; Hm_lvt_ba776f11bc6ba54ecefb321520244a3c=1525742554; Hm_ct_6bcd52f51e9b3dce32bec4a3997715ac=1788*1*PC_VC; kd_user_id=12e440dd-aea4-44a8-8fd2-c24deca406d8; __yadk_uid=gNmWave0JakcPLdUOtSrhh2X5U5xE1Xj; __utma=17226283.374548739.1525915028.1525915028.1525915028.1; __utmz=17226283.1525915028.1.1.utmcsr=csdn.net|utmccn=(referral)|utmcmd=referral|utmcct=/; smidV2=2018051116182222833b874e01d8817adf35439c5d8ad500ab066707269e310; dc_session_id=10_1526258539424.561571; TY_SESSION_ID=3ff47866-e046-4e70-b8ba-2eead6b87b32; Hm_lvt_6bcd52f51e9b3dce32bec4a3997715ac=1526261249,1526261822,1526261884,1526265579; Hm_lpvt_6bcd52f51e9b3dce32bec4a3997715ac=1526265579; dc_tos=p8p623",
    "User-Agent":"Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36"}

def parseUrl(url):
    response = requests.get(url, headers=headers)
    return response.content.decode()
