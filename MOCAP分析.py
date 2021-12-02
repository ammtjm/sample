import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#ディレクトリ指定
os.chdir(r"C:\Users\robor\PycharmProjects\pythonProject3\実験20211124\mocap")
#MOCAPデータ読込
df=pd.read_csv("20211124学生室実験5Char00.csv",sep=",",engine="python")
#MOCAPにはタイムスタンプがついてないのでCSVに時刻列を追加
for i in range(len(df.index)):
    df.loc[i,["time"]]=(i-1)/120
df.drop(df.index[0], inplace=True)
#開始時刻合わせ
#df=df[1630:2800]  #20211113tekubisotogawa5output.csv使用時
#df=df[1297:2430]    #20211113tekubiutigawa1Char00.csv
#df=df[1906:2881] #202110291218tekubiutigawayokohuri3tyokuritu10_1Char00.csv
#df=df[1600:2775]  #20211113tekubiutigawazyusinkibutukeru1Char00.csv
#df=df[397:1885] #20211122学生室歩行実験8Char00.csv
#df=df[588:1088]#20211122学生室歩行実験7Char00.csv
df_set=df.copy()
df_set=df_set[1090:1708]
#df_set=df[:]
df_set["time"]=(df_set["time"])-(df_set.iat[0,946])
#BLEデータ読込
rssi = []
clocktime = []
realtime = []
#ファイル読み込み
# データが格納されている作業ディレクトリまでパス指定
os.chdir(r"C:\Users\robor\PycharmProjects\pythonProject3\実験20211124\rssi")
df2 = pd.read_csv("get_rssi_20211124_1508学生室実験5.csv")

#1つの変数に格納

df2 = df2.dropna(how='any')
#df2["rssi"]=-df2["rssi"]


#開始時刻合わせにデータを切る
df2_set_a=df2.copy()
#df2=df2[3976:4951]    # get_rssi_20211113_1025tekubisotogawa4.csv使用時
#df2=df2[1644:2588]     #get_rssi_20211113_1034tekubiutigawa1.csv使用時
#df2=df2[2168:2980] #get_rssi_20211029_1253tekubiutigawayokohuri3.csv
#df2=df2[2090:3069]#get_rssi_20211113_1042tekubiutigawazyusinkibutukeru1.csv
#df2=df2[2038:3300]#get_rssi_20211122_1449学生室歩行実験8.csv
#df2=df2[1200:1800]#get_rssi_20211122_1446学生室歩行実験7.csv
df2_set_a=df2[3090:3600]
df2_set=df2_set_a.copy()
#df2_set=df2[:]
#時刻合わせ，ここでMOCAPとRSSIの計測開始時刻を合わせる

df2_set["realtime"]=(df2_set["clocktime"]/32768)-(df2_set.iat[0,1]/32768)
#plt.show()
#以下は補間作業
#等間隔でMOCAPデータのポイントを多くつくる
x_latent = np.linspace(min(df_set["time"]), max(df_set["time"]), len(df_set["time"])*100//120)
#rssi_latent=np.linspace(min(df2["realtime"]), max(df2["realtime"]), len(df2["realtime"]))
from scipy import interpolate
ip6 = ["2次スプライン補間", lambda x, y: interpolate.interp1d(x, y, kind="quadratic")]

for method_name, method in [ip6]:
    print(method_name)
    plt.subplot(3,1,1)
    #fitted_curve = method(df["time"], df["15-X-y"])
    #fitted_curve2=method(df2["realtime"],df2["rssi"])
    fitted_curve = method(df_set["time"], df_set["11-X-y"])
    #plt.plot(df["time"], df["15-X-y"], label="observed")
    plt.plot(df_set["time"], df_set["11-X-y"], label="observed")
    #等間隔の補間された時間とそれに合わせた関数のプロット
    plt.scatter(x_latent, fitted_curve(x_latent), c="red", label="fitted")
    plt.grid()
    plt.legend()

    plt.subplot(3,1,2)
    #plt.scatter(df["time"], df["15-X-y"], label="observed")
    plt.scatter(df_set["time"], df_set["11-X-y"], label="observed")
    plt.xlim(1,1.4)
    plt.grid()
    plt.legend()

    plt.subplot(3,1,3)
    plt.scatter(x_latent, fitted_curve(x_latent), c="red", label="fitted")
    plt.xlim(1,1.4)
    plt.grid()
    plt.legend()
    plt.show()

#相互相関関数の計算
#まずは正規化(平均0,標準偏差1)処理を行う
normalize_x = (fitted_curve(x_latent) - fitted_curve(x_latent).mean())/np.std(fitted_curve(x_latent))
normalize_rssi = (df2_set["rssi"] - df2_set["rssi"].mean())/np.std(df2_set["rssi"])
#normalize_rssi = (fitted_curve2(rssi_latent) - fitted_curve2(rssi_latent).mean())/np.std(fitted_curve2(rssi_latent))
#畳み込み演算,両関数の積の最大値を探している,http://www.slp.k.hosei.ac.jp/~itou/lecture/2011/DigitalData/06_text.pdf
#corr = np.correlate(normalize_rssi, normalize_x, "full")/len(normalize_x)
corr = np.correlate(normalize_x, normalize_rssi, "full")/len(normalize_rssi)
#遅れの計算, corr.argmaxは相互相関の最大値
estimated_delay = corr.argmax() - (len(normalize_x) - 1)
#1行0.01sなので遅れた時間は以下のようになる
delta_T = estimated_delay*0.01
print("estimated delay is " + str(estimated_delay))
print("delta_T is "+str(delta_T))
print(max(abs(corr)))
#グラフパート
#MOCAPのプロット
plt.subplot(4, 1, 1)
plt.ylabel("11-X-y[m]")
#plt.xlim(0,10)
plt.plot(x_latent, fitted_curve(x_latent), c="red", label="fitted")
plt.grid(which = "major", axis = "both", color = "blue", alpha = 0.8,linestyle = "--", linewidth = 1)
#BLEのプロット
plt.subplot(4, 1, 2)
#plt.xlim(0,10)
plt.ylabel("RSSI[dB]")
plt.plot(df2_set["realtime"], df2_set["rssi"],color="g")
plt.grid(which = "major", axis = "both", color = "blue", alpha = 0.8,linestyle = "--", linewidth = 1)
#2つの波形がどれだけズレているのか確認
plt.subplot(4, 1, 3)
plt.xlim(0,10)
plt.ylabel("confirm_shift")
plt.plot(x_latent, normalize_x, alpha=0.5)
plt.plot(df2_set["realtime"], normalize_rssi, alpha=0.5)
plt.grid(which = "major", axis = "both", color = "blue", alpha = 0.8,linestyle = "--", linewidth = 1)

#delta_Tだけずらした結果の確認
plt.subplot(4, 1, 4)
plt.xlim(0,10)
plt.ylabel("fit")
#MOCAP
plt.plot(x_latent, normalize_x, alpha=0.5)
#BLE
#rssiが遅れているという前提
plt.plot(df2_set["realtime"]+delta_T, normalize_rssi, alpha=0.5)
plt.grid(which = "major", axis = "both", color = "blue", alpha = 0.8,linestyle = "--", linewidth = 1)

"""
plt.subplot(5, 1, 5)
plt.ylabel("corr")
plt.plot(np.arange(len(corr)) - len(df2["realtime"]) + 1, corr, color="r")
plt.xlim([0, len(fitted_curve(x_latent))])
"""
plt.show()





#以下から機械学習
#1データの取得時刻を合わせ, 入出力の数を揃える
#2データを訓練用・テスト用に分割
#3訓練用データを使用して機械学習
#データ範囲を指定, スタートは同期の開始点
df_syn_set=df[1090:6150]
df_syn=df_syn_set.copy()
df2_syn_set=df2[3090:8000]

#rssiとmocapの補間の取得時刻を揃えるためにrssiの補間に使う時間の長いデータを用意
#rssiの開始時刻だけデータ範囲の始点と揃えて，それ以降のデータ全てを用意する
df2_timesetting=df2[3090:]
df2_timesetting2=df2_timesetting.copy()
df2_timesetting_true=df2_timesetting.copy()
df2_timesetting_true["realtime"]=(df2_timesetting2["clocktime"]/32768)-(df2_timesetting2.iat[0,1]/32768)
df2_time=df2_timesetting_true.copy()
df2_time=df2_time+delta_T
df2_timesetting_true["realtime"]=df2_timesetting_true["realtime"]+delta_T
df2_syn=df2_syn_set.copy()
df2_syn["rssi"]=-df2_syn["rssi"]
#mocapとbleの時刻設定

df_syn["time"]=(df_syn["time"])-(df_syn.iat[0,946])
df2_syn["realtime"]=(df2_syn["clocktime"]/32768)-(df2_syn.iat[0,1]/32768)
df2_syn["realtime"]=df2_syn["realtime"]+delta_T

#以下は補間作業
#等間隔でMOCAPデータとBLEデータの点数を揃える
x_latent = np.linspace(min(df2_syn["realtime"]), max(df_syn["time"]), len(df_syn["time"]))
#x_latent=np.linspace(min(df2_syn["realtime"]), max(df2_syn["realtime"]), len(df2_syn["realtime"]))
from scipy import interpolate
ip6 = ["2次スプライン補間", lambda x, y: interpolate.interp1d(x, y, kind="quadratic")]
#946列分の補間のために空の行列を用意
normalize_mocap_interpolation=[]

#補間作業
for method_name, method in [ip6]:
    #946列分それぞれに補間を作る
    for j in range(len(df_syn.columns)-1):
        #squeeze()で1次元の配列にすることで補間のmethodに使用可能にしている
        fitted_curve_mocap = method(df_syn["time"], df_syn.iloc[:,lambda df_syn:[j]].squeeze())
        #補間で作り直す間隔はBLEと揃えるためにx_latentを使用
        #mocap標準化（standardization）, 平均0,分散1にしてる
        with np.errstate(all="ignore"):
            normalize_mocap_interpolation.append((fitted_curve_mocap(x_latent) - fitted_curve_mocap(x_latent).mean()) / np.std(fitted_curve_mocap(x_latent)))
    #rssiの補間はmocapの時間よりも長い者を用意
    fitted_curve_rssi = method(df2_time["realtime"],df2_time["rssi"])
#rssi標準化, 取得時刻/間隔はMOCAPと同期させるためにx_latentを用いた
normalize_rssi_interpolation = (fitted_curve_rssi(x_latent) - fitted_curve_rssi(x_latent).mean())/np.std(fitted_curve_rssi(x_latent))
#リスト型になったmocap,rssiデータをdataframeに変換
normalize_mocap_interpolation=pd.DataFrame(normalize_mocap_interpolation).T
normalize_rssi_interpolation=-pd.Series(normalize_rssi_interpolation).T
#列名をつける
normalize_mocap_interpolation.columns=["01-X-x",	"01-X-y",	"01-X-z",	"01-V-x",	"01-V-y",	"01-V-z",	"01-Q-s",	"01-Q-x"	,"01-Q-y"\
        ,"01-Q-z",	"01-A-x",	"01-A-y",	"01-A-z",	"01-W-x",	"01-W-y",	"01-W-z",	"02-X-x",	"02-X-y",	"02-X-z"\
        ,"02-V-x",	"02-V-y",	"02-V-z",	"02-Q-s",	"02-Q-x",	"02-Q-y",	"02-Q-z",	"02-A-x",	"02-A-y",	"02-A-z"\
    ,"02-W-x",	"02-W-y",	"02-W-z",	"03-X-x",	"03-X-y",	"03-X-z",	"03-V-x",	"03-V-y",	"03-V-z",	"03-Q-s"\
    ,	"03-Q-x",	"03-Q-y",	"03-Q-z",	"03-A-x",	"03-A-y",	"03-A-z",	"03-W-x",	"03-W-y",	"03-W-z"\
        ,"04-X-x",	"04-X-y",	"04-X-z",	"04-V-x",	"04-V-y",	"04-V-z",	"04-Q-s",	"04-Q-x",	"04-Q-y",	"04-Q-z"\
        ,"04-A-x",	"04-A-y",	"04-A-z",	"04-W-x"	,"04-W-y",	"04-W-z",	"05-X-x",	"05-X-y",	"05-X-z",	"05-V-x"\
        ,"05-V-y",	"05-V-z",	"05-Q-s",	"05-Q-x",	"05-Q-y",	"05-Q-z",	"05-A-x",	"05-A-y",	"05-A-z",	"05-W-x"\
        ,"05-W-y",	"05-W-z",	"06-X-x",	"06-X-y",	"06-X-z",	"06-V-x",	"06-V-y",	"06-V-z",	"06-Q-s",	"06-Q-x"\
        ,"06-Q-y",	"06-Q-z",	"06-A-x",	"06-A-y",	"06-A-z",	"06-W-x",	"06-W-y",	"06-W-z",	"07-X-x",	"07-X-y"\
        ,"07-X-z",	"07-V-x",	"07-V-y",	"07-V-z",	"07-Q-s",	"07-Q-x",	"07-Q-y",	"07-Q-z",	"07-A-x",	"07-A-y"\
        ,"07-A-z",	"07-W-x",	"07-W-y",	"07-W-z",	"08-X-x",	"08-X-y",	"08-X-z",	"08-V-x",	"08-V-y",	"08-V-z"\
        ,"08-Q-s",	"08-Q-x",	"08-Q-y",	"08-Q-z",	"08-A-x",	"08-A-y",	"08-A-z",	"08-W-x",	"08-W-y",	"08-W-z"\
        ,"09-X-x",	"09-X-y",	"09-X-z",	"09-V-x",	"09-V-y",	"09-V-z",	"09-Q-s",	"09-Q-x",	"09-Q-y",	"09-Q-z"\
        ,"09-A-x",	"09-A-y",	"09-A-z",	"09-W-x",	"09-W-y",	"09-W-z",	"10-X-x",	"10-X-y",	"10-X-z",	"10-V-x"\
        ,"10-V-y",	"10-V-z",	"10-Q-s",	"10-Q-x",	"10-Q-y",	"10-Q-z",	"10-A-x",	"10-A-y",	"10-A-z",	"10-W-x"\
        ,"10-W-y",	"10-W-z",	"11-X-x",	"11-X-y",	"11-X-z",	"11-V-x",	"11-V-y",	"11-V-z",	"11-Q-s",	"11-Q-x"\
        ,"11-Q-y",	"11-Q-z",	"11-A-x",	"11-A-y",	"11-A-z",	"11-W-x",	"11-W-y",	"11-W-z",	"12-X-x",	"12-X-y"\
        ,"12-X-z",	"12-V-x",	"12-V-y",	"12-V-z",	"12-Q-s",	"12-Q-x",	"12-Q-y",	"12-Q-z",	"12-A-x",	"12-A-y"\
        ,"12-A-z",	"12-W-x",	"12-W-y",	"12-W-z",	"13-X-x",	"13-X-y",	"13-X-z",	"13-V-x",	"13-V-y",	"13-V-z"\
        ,"13-Q-s",	"13-Q-x",	"13-Q-y",	"13-Q-z",	"13-A-x",	"13-A-y",	"13-A-z",	"13-W-x",	"13-W-y",	"13-W-z"\
        ,"14-X-x",	"14-X-y",	"14-X-z",	"14-V-x",	"14-V-y",	"14-V-z",	"14-Q-s",	"14-Q-x",	"14-Q-y",	"14-Q-z"\
        ,"14-A-x",	"14-A-y",	"14-A-z",	"14-W-x",	"14-W-y",	"14-W-z",	"15-X-x",	"15-X-y",	"15-X-z",	"15-V-x"\
        ,"15-V-y",	"15-V-z",	"15-Q-s",	"15-Q-x",	"15-Q-y",	"15-Q-z",	"15-A-x",	"15-A-y",	"15-A-z",	"15-W-x"\
        ,"15-W-y",	"15-W-z",	"16-X-x",	"16-X-y",	"16-X-z",	"16-V-x",	"16-V-y",	"16-V-z",	"16-Q-s",	"16-Q-x"\
        ,"16-Q-y",	"16-Q-z",	"16-A-x",	"16-A-y",	"16-A-z",	"16-W-x",	"16-W-y",	"16-W-z",	"17-X-x",	"17-X-y"\
        ,"17-X-z",	"17-V-x",	"17-V-y",	"17-V-z",	"17-Q-s",	"17-Q-x",	"17-Q-y",	"17-Q-z",	"17-A-x",	"17-A-y"\
        ,"17-A-z",	"17-W-x",	"17-W-y",	"17-W-z",	"18-X-x",	"18-X-y",	"18-X-z",	"18-V-x",	"18-V-y",	"18-V-z"\
        ,"18-Q-s",	"18-Q-x",	"18-Q-y",	"18-Q-z",	"18-A-x",	"18-A-y",	"18-A-z",	"18-W-x",	"18-W-y",	"18-W-z"\
        ,"19-X-x",	"19-X-y",	"19-X-z",	"19-V-x",	"19-V-y",	"19-V-z",	"19-Q-s",	"19-Q-x",	"19-Q-y",	"19-Q-z"\
        ,"19-A-x",	"19-A-y",	"19-A-z",	"19-W-x",	"19-W-y",	"19-W-z",	"20-X-x",	"20-X-y",	"20-X-z",	"20-V-x"\
        ,"20-V-y",	"20-V-z",	"20-Q-s",	"20-Q-x",	"20-Q-y",	"20-Q-z",	"20-A-x",	"20-A-y",	"20-A-z",	"20-W-x"\
        ,"20-W-y",	"20-W-z",	"21-X-x",	"21-X-y",	"21-X-z",	"21-V-x",	"21-V-y",	"21-V-z",	"21-Q-s",	"21-Q-x"\
        ,"21-Q-y",	"21-Q-z",	"21-A-x",	"21-A-y",	"21-A-z",	"21-W-x",	"21-W-y",	"21-W-z",	"contactL",	"contactR"\
        ,"22-X-x",	"22-X-y",	"22-X-z",	"22-V-x",	"22-V-y",	"22-V-z",	"22-Q-s",	"22-Q-x",	"22-Q-y",	"22-Q-z"\
        ,"22-A-x",	"22-A-y",	"22-A-z",	"22-W-x",	"22-W-y",	"22-W-z",	"23-X-x",	"23-X-y",	"23-X-z",	"23-V-x"\
        ,"23-V-y",	"23-V-z",	"23-Q-s",	"23-Q-x",	"23-Q-y",	"23-Q-z",	"23-A-x",	"23-A-y",	"23-A-z",	"23-W-x"\
        ,"23-W-y",	"23-W-z",	"24-X-x",	"24-X-y",	"24-X-z",	"24-V-x",	"24-V-y",	"24-V-z",	"24-Q-s",	"24-Q-x"\
        ,"24-Q-y",	"24-Q-z",	"24-A-x",	"24-A-y",	"24-A-z",	"24-W-x",	"24-W-y",	"24-W-z",	"25-X-x",	"25-X-y"\
        ,"25-X-z",	"25-V-x",	"25-V-y",	"25-V-z",	"25-Q-s",	"25-Q-x",	"25-Q-y",	"25-Q-z",	"25-A-x",	"25-A-y"\
        ,"25-A-z",	"25-W-x",	"25-W-y",	"25-W-z",	"26-X-x",	"26-X-y",	"26-X-z",	"26-V-x",	"26-V-y",	"26-V-z"\
        ,"26-Q-s",	"26-Q-x",	"26-Q-y",	"26-Q-z",	"26-A-x",	"26-A-y",	"26-A-z",	"26-W-x",	"26-W-y",	"26-W-z"\
        ,"27-X-x",	"27-X-y",	"27-X-z",	"27-V-x",	"27-V-y",	"27-V-z",	"27-Q-s",	"27-Q-x",	"27-Q-y",	"27-Q-z"\
        ,"27-A-x",	"27-A-y",	"27-A-z",	"27-W-x",	"27-W-y",	"27-W-z",	"28-X-x",	"28-X-y",	"28-X-z",	"28-V-x"\
        ,"28-V-y",	"28-V-z",	"28-Q-s",	"28-Q-x",	"28-Q-y",	"28-Q-z",	"28-A-x",	"28-A-y",	"28-A-z",	"28-W-x"\
        ,"28-W-y",	"28-W-z",	"29-X-x",	"29-X-y",	"29-X-z",	"29-V-x",	"29-V-y",	"29-V-z",	"29-Q-s",	"29-Q-x"\
        ,"29-Q-y",	"29-Q-z",	"29-A-x",	"29-A-y",	"29-A-z",	"29-W-x",	"29-W-y",	"29-W-z",	"30-X-x",	"30-X-y"\
        ,"30-X-z",	"30-V-x",	"30-V-y",	"30-V-z",	"30-Q-s",	"30-Q-x",	"30-Q-y",	"30-Q-z",	"30-A-x",	"30-A-y"\
        ,"30-A-z",	"30-W-x",	"30-W-y",	"30-W-z",	"31-X-x",	"31-X-y",	"31-X-z",	"31-V-x",	"31-V-y",	"31-V-z"\
        ,"31-Q-s",	"31-Q-x",	"31-Q-y",	"31-Q-z",	"31-A-x",	"31-A-y",	"31-A-z",	"31-W-x",	"31-W-y",	"31-W-z"\
        ,"32-X-x",	"32-X-y",	"32-X-z",	"32-V-x",	"32-V-y",	"32-V-z",	"32-Q-s",	"32-Q-x",	"32-Q-y",	"32-Q-z"\
        ,"32-A-x",	"32-A-y",	"32-A-z",	"32-W-x",	"32-W-y",	"32-W-z",	"33-X-x",	"33-X-y",	"33-X-z",	"33-V-x"\
        ,"33-V-y",	"33-V-z",	"33-Q-s",	"33-Q-x",	"33-Q-y",	"33-Q-z",	"33-A-x",	"33-A-y",	"33-A-z",	"33-W-x"\
        ,"33-W-y",	"33-W-z",	"34-X-x",	"34-X-y",	"34-X-z",	"34-V-x",	"34-V-y",	"34-V-z",	"34-Q-s",	"34-Q-x"\
        ,"34-Q-y",	"34-Q-z",	"34-A-x",	"34-A-y",	"34-A-z",	"34-W-x",	"34-W-y",	"34-W-z",	"35-X-x",	"35-X-y"\
        ,"35-X-z",	"35-V-x",	"35-V-y",	"35-V-z",	"35-Q-s",	"35-Q-x",	"35-Q-y",	"35-Q-z",	"35-A-x",	"35-A-y"\
        ,"35-A-z",	"35-W-x",	"35-W-y",	"35-W-z",	"36-X-x",	"36-X-y",	"36-X-z",	"36-V-x",	"36-V-y",	"36-V-z"\
        ,"36-Q-s",	"36-Q-x",	"36-Q-y",	"36-Q-z",	"36-A-x",	"36-A-y",	"36-A-z",	"36-W-x",	"36-W-y",	"36-W-z"\
        ,"37-X-x",	"37-X-y",	"37-X-z",	"37-V-x",	"37-V-y",	"37-V-z",	"37-Q-s",	"37-Q-x",	"37-Q-y",	"37-Q-z"\
        ,"37-A-x",	"37-A-y",	"37-A-z",	"37-W-x",	"37-W-y",	"37-W-z",	"38-X-x",	"38-X-y",	"38-X-z",	"38-V-x"\
        ,"38-V-y",	"38-V-z",	"38-Q-s",	"38-Q-x",	"38-Q-y",	"38-Q-z",	"38-A-x",	"38-A-y",	"38-A-z",	"38-W-x"\
        ,"38-W-y",	"38-W-z",	"39-X-x",	"39-X-y",	"39-X-z",	"39-V-x",	"39-V-y",	"39-V-z",	"39-Q-s",	"39-Q-x"\
        ,"39-Q-y",	"39-Q-z",	"39-A-x",	"39-A-y",	"39-A-z",	"39-W-x",	"39-W-y",	"39-W-z",	"40-X-x",	"40-X-y"\
        ,"40-X-z",	"40-V-x",	"40-V-y",	"40-V-z",	"40-Q-s",	"40-Q-x",	"40-Q-y",	"40-Q-z",	"40-A-x",	"40-A-y"\
        ,"40-A-z",	"40-W-x",	"40-W-y",	"40-W-z",	"41-X-x",	"41-X-y",	"41-X-z",	"41-V-x",	"41-V-y",	"41-V-z"\
        ,"41-Q-s",	"41-Q-x",	"41-Q-y",	"41-Q-z",	"41-A-x",	"41-A-y",	"41-A-z",	"41-W-x",	"41-W-y",	"41-W-z"\
        ,"42-X-x",	"42-X-y",	"42-X-z",	"42-V-x",	"42-V-y",	"42-V-z",	"42-Q-s",	"42-Q-x",	"42-Q-y",	"42-Q-z"\
        ,"42-A-x",	"42-A-y",	"42-A-z",	"42-W-x",	"42-W-y",	"42-W-z",	"43-X-x",	"43-X-y",	"43-X-z",	"43-V-x"\
        ,"43-V-y",	"43-V-z",	"43-Q-s",	"43-Q-x",	"43-Q-y",	"43-Q-z",	"43-A-x",	"43-A-y",	"43-A-z",	"43-W-x"\
        ,"43-W-y",	"43-W-z",	"44-X-x",	"44-X-y",	"44-X-z",	"44-V-x",	"44-V-y",	"44-V-z",	"44-Q-s",	"44-Q-x"\
        ,"44-Q-y",	"44-Q-z",	"44-A-x",	"44-A-y",	"44-A-z",	"44-W-x",	"44-W-y",	"44-W-z",	"45-X-x",	"45-X-y"\
        ,"45-X-z",	"45-V-x",	"45-V-y",	"45-V-z",	"45-Q-s",	"45-Q-x",	"45-Q-y",	"45-Q-z",	"45-A-x",	"45-A-y"\
        ,"45-A-z",	"45-W-x",	"45-W-y",	"45-W-z",	"46-X-x",	"46-X-y",	"46-X-z",	"46-V-x",	"46-V-y",	"46-V-z"\
        ,"46-Q-s",	"46-Q-x",	"46-Q-y",	"46-Q-z",	"46-A-x",	"46-A-y",	"46-A-z",	"46-W-x",	"46-W-y",	"46-W-z"\
        ,"47-X-x",	"47-X-y",	"47-X-z",	"47-V-x",	"47-V-y",	"47-V-z",	"47-Q-s",	"47-Q-x",	"47-Q-y",	"47-Q-z"\
        ,"47-A-x",	"47-A-y",	"47-A-z",	"47-W-x",	"47-W-y",	"47-W-z",	"48-X-x",	"48-X-y",	"48-X-z",	"48-V-x"\
        ,"48-V-y",	"48-V-z",	"48-Q-s",	"48-Q-x",	"48-Q-y",	"48-Q-z",	"48-A-x",	"48-A-y",	"48-A-z",	"48-W-x"\
        ,"48-W-y",	"48-W-z",	"49-X-x",	"49-X-y",	"49-X-z",	"49-V-x",	"49-V-y",	"49-V-z",	"49-Q-s",	"49-Q-x"\
        ,"49-Q-y",	"49-Q-z",	"49-A-x",	"49-A-y",	"49-A-z",	"49-W-x",	"49-W-y",	"49-W-z",	"50-X-x",	"50-X-y"\
        ,"50-X-z",	"50-V-x",	"50-V-y",	"50-V-z",	"50-Q-s",	"50-Q-x",	"50-Q-y",	"50-Q-z",	"50-A-x",	"50-A-y"\
        ,"50-A-z",	"50-W-x",	"50-W-y",	"50-W-z",	"51-X-x",	"51-X-y",	"51-X-z",	"51-V-x",	"51-V-y",	"51-V-z"\
        ,"51-Q-s",	"51-Q-x",	"51-Q-y",	"51-Q-z",	"51-A-x",	"51-A-y",	"51-A-z",	"51-W-x",	"51-W-y",	"51-W-z"\
        ,"52-X-x",	"52-X-y",	"52-X-z",	"52-V-x",	"52-V-y",	"52-V-z",	"52-Q-s",	"52-Q-x",	"52-Q-y",	"52-Q-z"\
        ,"52-A-x",	"52-A-y",	"52-A-z",	"52-W-x",	"52-W-y",	"52-W-z",	"53-X-x",	"53-X-y",	"53-X-z",	"53-V-x"\
        ,"53-V-y",	"53-V-z",	"53-Q-s",	"53-Q-x",	"53-Q-y",	"53-Q-z",	"53-A-x",	"53-A-y",	"53-A-z",	"53-W-x"\
        ,"53-W-y",	"53-W-z",	"54-X-x",	"54-X-y",	"54-X-z",	"54-V-x",	"54-V-y",	"54-V-z",	"54-Q-s",	"54-Q-x"\
        ,"54-Q-y",	"54-Q-z",	"54-A-x",	"54-A-y",	"54-A-z",	"54-W-x",	"54-W-y",	"54-W-z",	"55-X-x",	"55-X-y"\
        ,"55-X-z",	"55-V-x",	"55-V-y",	"55-V-z",	"55-Q-s",	"55-Q-x",	"55-Q-y",	"55-Q-z",	"55-A-x",	"55-A-y"\
        ,"55-A-z",	"55-W-x",	"55-W-y",	"55-W-z",	"56-X-x",	"56-X-y",	"56-X-z",	"56-V-x",	"56-V-y",	"56-V-z"\
        ,"56-Q-s",	"56-Q-x",	"56-Q-y",	"56-Q-z",	"56-A-x",	"56-A-y",	"56-A-z",	"56-W-x",	"56-W-y",	"56-W-z"\
        ,"57-X-x",	"57-X-y",	"57-X-z",	"57-V-x",	"57-V-y",	"57-V-z",	"57-Q-s",	"57-Q-x",	"57-Q-y",	"57-Q-z"\
        ,"57-A-x",	"57-A-y",	"57-A-z",	"57-W-x",	"57-W-y",	"57-W-z",	"58-X-x",	"58-X-y",	"58-X-z",	"58-V-x"\
        ,"58-V-y",	"58-V-z",	"58-Q-s",	"58-Q-x",	"58-Q-y",	"58-Q-z",	"58-A-x",	"58-A-y",	"58-A-z",	"58-W-x"\
        ,"58-W-y",	"58-W-z",	"59-X-x",	"59-X-y",	"59-X-z",	"59-V-x",	"59-V-y",	"59-V-z",	"59-Q-s",	"59-Q-x"\
        ,"59-Q-y",	"59-Q-z",	"59-A-x",	"59-A-y",	"59-A-z",	"59-W-x",	"59-W-y",	"59-W-z"]
#相互相関で同期し, 2次スプラインで補間された両時系列データとなっているか確認する
#RSSIのグラフはピークがmocapと一致しているか確認させるために反転させている
plt.figure()
plt.plot(x_latent, normalize_mocap_interpolation["11-X-y"], alpha=0.5)
plt.plot(x_latent, -normalize_rssi_interpolation, alpha=0.5)
plt.show()

#訓練データとテストデータに分割する?

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa import stattools
from statsmodels.tsa.api import VAR
import seaborn as sns

#2021/12/02 帰ったらXをsplitした後にdiffをするように変更する, でないとinverting transformationが間違えたまま

#ここからVAR
x_latent=pd.Series(x_latent).T
df_concat = pd.concat([normalize_mocap_interpolation,normalize_rssi_interpolation],axis=1)
df_concat_new= df_concat.rename(columns={0:"rssi"})
X_before=df_concat_new.copy()
#指先のデータなどの欠損値を削除, VARは定常データしか扱えないので1次の階差をとる
#X_before_b= X_before.diff(axis=0)[1:].dropna(axis=1, how="any")
#print(X_before_b.shape)
X_before_concat=pd.concat([X_before, x_latent], axis=1)
X=X_before_concat.rename(columns={0:"time"}).dropna(axis=1, how="any")
print(list(X.columns))
#X=X.set_index("time")
X=pd.DataFrame(X)
#時間をインデックスに割り当てる
#df_mocap_rssi_new_1=df_mocap_rssi_new.set_index("time")
#指先のデータなどの欠損値を削除
#print(list(X))
dec_testsize=10
#実験データは21までを使用する, それ以上の項目は今回の実験では取得されていない
#16,19,20,21のA,Wが0に近いためかコレスキー分解の平方根が負の値になりエラー, 位置情報のみを使ったところ上手くいった
X_new = X[['01-X-x', '01-X-y','01-X-z' , '02-X-x', '02-X-y', '02-X-z','03-X-x', '03-X-y', '03-X-z','04-X-x', '04-X-y', '04-X-z','05-X-x', '05-X-y', '05-X-z','06-X-x', '06-X-y', '06-X-z','07-X-x', '07-X-y', '07-X-z','08-X-x', '08-X-y', '08-X-z','09-X-x', '09-X-y', '09-X-z','10-X-x', '10-X-y', '10-X-z','11-X-x', '11-X-y', '11-X-z','12-X-x', '12-X-y', '12-X-z','13-X-x', '13-X-y', '13-X-z','14-X-x', '14-X-y', '14-X-z','15-X-x', '15-X-y', '15-X-z','16-X-x', '16-X-y', '16-X-z','17-X-x', '17-X-y', '17-X-z','18-X-x', '18-X-y', '18-X-z','19-X-x', '19-X-y', '19-X-z','20-X-x', '20-X-y', '20-X-z','21-X-x', '21-X-y', '21-X-z',"rssi"]]
#ここから参考文献, https://ichi.pro/var-to-vecm-o-shiyoshita-toki-keiretsu-bunseki-kanzenna-python-ko-do-o-shiyoshita-tokeiteki-apuro-chi-183196183082520


X_train, X_test = X_new[0:-dec_testsize], X_new[-dec_testsize:]
X_train_diff=X_train.diff(axis=0)[1:].dropna(axis=1, how="any")

model = VAR(X_train_diff)

#Fit to a VAR model
model_fit = model.fit(maxlags=15,ic='bic')
#Print a summary of the model results
model_fit.summary()
# Get the lag order
lag_order = model_fit.k_ar
print(lag_order)
"""
for i in range (len(X_train_diff.columns)):
    adf = stattools.adfuller(X_train_diff.iloc[:,i], regression='ctt')
    if -3.83>adf[0]:
        print(X_train_diff.columns[i])
        print('t値 : {:.2f}, p値 : {:.1f}%'.format(adf[0], adf[1]*100))
        #print('データ数 : {}, 使用されたラグ数 : {}'.format(adf[3], adf[2]))
        print('検定統計量における棄却値 : 1%={:.2f}, 5%={:.2f}, 10%={:.2f}'.format(
            adf[4]['1%'], adf[4]['5%'], adf[4]['10%']))
"""
# Input data for forecasting
input_data = X_train_diff.values[-lag_order:]
# forecasting
pred = model_fit.forecast(y=input_data, steps=dec_testsize)
pred = (pd.DataFrame(pred, index=X_test.index, columns=X_test.columns + '_pred'))
#print(pred.shape)

# inverting transformation 元に戻す
def invert_transformation(X_train, pred_df):
  forecast = pred.copy()
  columns = X_train.columns
  for col in columns:
        forecast[str(col)+'_pred'] = X_train[col].iloc[-1] + forecast[str(col) +'_pred'].cumsum()
        return forecast
output = invert_transformation(X_train, pred)
#print(output)

#Calculate forecast bias
forecast_errors = X_test.loc[:, 'rssi']- output.loc[:,'rssi_pred']
bias = sum(forecast_errors) * 1.0/len(X_test['rssi'])
print('Bias: %f' % bias)

#Calculate mean absolute error
mae = mean_absolute_error(X_test['rssi'],output['rssi_pred'])
print('MAE: %f' % mae)
#Calculate mean squared error and root mean squared error
mse = mean_squared_error(X_test['rssi'], output['rssi_pred'])
print('MSE: %f' % mse)
rmse = np.sqrt(mse)
print('RMSE: %f' % rmse)


#LSTM



