import base64
import random
import re
import time

import cv2
import numpy as np
import requests
from loguru import logger
if __name__=="__main__":

    num = 1000
    success=0
    fail=0
    for i in range(num):
        cookies = {
            'XSRF-TOKEN': '2570367b-c4cb-4583-ab98-99f2c012fe3d',
            '_samesite_flag_': 'true',
            'cookie2': '182efacad0b9b75a41d11c7c251ae6e4',
            't': 'c689c660b39b13324bf41e79176a4ee3',
            '_tb_token_': 'f57e537de6e88',
            'cna': 'xCM6H+A6cWwCAbcggskaGkBK',
            'arms_uid': '76a87a64-0046-4fb5-ac5a-ed20a644d531',
            '_bl_uid': 'bpljyzRIjk4mzn26y8qI73y3p0h0',
            '3PcFlag': '1723020837872',
            'x5sectag': '565868',
            'isg': 'BENDsF1xIRk3Ru1GvNgfeT4g0gftuNf6OmCl5XUg56IZNGJW_ItYSwpiqsR6lC_y',
        }

        headers = {
            'accept': '*/*',
            'accept-language': 'zh-CN,zh;q=0.9',
            'cache-control': 'no-cache',
            # 'cookie': 'XSRF-TOKEN=2570367b-c4cb-4583-ab98-99f2c012fe3d; _samesite_flag_=true; cookie2=182efacad0b9b75a41d11c7c251ae6e4; t=c689c660b39b13324bf41e79176a4ee3; _tb_token_=f57e537de6e88; cna=xCM6H+A6cWwCAbcggskaGkBK; arms_uid=76a87a64-0046-4fb5-ac5a-ed20a644d531; _bl_uid=bpljyzRIjk4mzn26y8qI73y3p0h0; 3PcFlag=1723020837872; x5sectag=565868; isg=BENDsF1xIRk3Ru1GvNgfeT4g0gftuNf6OmCl5XUg56IZNGJW_ItYSwpiqsR6lC_y',
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'referer': 'https://havanalogin.taobao.com//newlogin/account/check.do/_____tmd_____/punish?x5secdata=xde92d862272f9972eaa6954db38d6a87efc956cbf74d2471c1723020842a-717315356a1780105780abaac3dl13794888912kcapslidev2__bx__havanalogin.taobao.com%3A443%2Fnewlogin%2Faccount%2Fcheck.do&x5step=2&action=captchacapslidev2&pureCaptcha=true',
            'sec-ch-ua': '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
        }

        params = {
            'token': 'aa6954db38d6a87efc956cbf74d2471c',
            'appKey': 'X82Y__c6ecc3ec0adcc6d4382d92be1710db07',
            'x5secdata': 'xde92d862272f9972eaa6954db38d6a87efc956cbf74d2471c1723020842a-717315356a1780105780abaac3dl13794888912kcapslidev2__bx__havanalogin.taobao.com:443/newlogin/account/check.do',
            'language': 'cn',
            'v': '008156039459877773',
        }

        response = requests.get(
            'https://havanalogin.taobao.com/newlogin/account/check.do/_____tmd_____/newslidecaptcha',
            params=params,
            cookies=cookies,
            headers=headers,
        )

        reqjson = {
                "title_img1":response.json()['data']['ques'],
                "background_img":response.json()['data']['imageData']
        }

        print(reqjson)

        response=requests.post("http://127.0.0.1:5000/getdetectresult",json=reqjson)
        response=response.json()
        print(response)

        if response['code']==0:
            success+=1
        else:
            fail+=1
        print(f"success:{success} fail:{fail} rate:{success/(success+fail)}")

        time.sleep(2)


        # resultx=response['data']['x']
        # img=base64.b64decode(res['imageData'].split('base64,')[-1])
        # img=cv2.imdecode(np.frombuffer(img,np.uint8),cv2.IMREAD_COLOR)
        #
        #
        # print(resultx,img.shape)
        # cv2.line(img,(resultx,0),(resultx,img.shape[0]),(0,0,255),2)
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
