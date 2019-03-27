> foo=read.table('~/Desktop/test.iou20.csv', sep=',')
> bar = as.numeric(foo)

# test 20
> mean(bar[1:150]); sd(bar[1:150]) # 16
[1] 0.264105
[1] 0.1180043
> mean(bar[150:300]); sd(bar[150:300]) # 17
[1] 0.2358283
[1] 0.09331188
> mean(bar[300:450]); sd(bar[300:450]) # 18 
[1] 0.2209759
[1] 0.1379925
> mean(bar[450:600]); sd(bar[450:600]) # 19
[1] 0.09986578
[1] 0.09713121
> mean(bar[600:750]); sd(bar[600:750]) # 20
[1] 0.214326
[1] 0.09360225

> foo=read.table('~/Desktop/test.iou15.csv', sep=',')
> bar = as.numeric(foo)
> mean(bar[1:150]); sd(bar[1:150]) # 16
[1] 0.2193204
[1] 0.1049463
> mean(bar[150:300]); sd(bar[150:300]) # 17
[1] 0.1965765
[1] 0.08682248
> mean(bar[300:450]); sd(bar[300:450]) # 18
[1] 0.1985902
[1] 0.1305335
> mean(bar[450:600]); sd(bar[450:600]) # 19
[1] 0.0772819
[1] 0.07993372
> mean(bar[600:750]); sd(bar[600:750]) # 20
[1] 0.1896303
[1] 0.08959316


> foo=read.table('~/Desktop/test.iou10.csv', sep=',')
> bar = as.numeric(foo)
> mean(bar[1:150]); sd(bar[1:150]) # 16
[1] 0.166004
[1] 0.09524608
> mean(bar[150:300]); sd(bar[150:300]) # 17
[1] 0.1443083
[1] 0.07526396
> mean(bar[300:450]); sd(bar[300:450]) # 18
[1] 0.1446855
[1] 0.114401
> mean(bar[450:600]); sd(bar[450:600]) # 19
[1] 0.05064124
[1] 0.05817133
> mean(bar[600:750]); sd(bar[600:750]) # 20
[1] 0.1419018
[1] 0.07509613


> foo=read.table('~/Desktop/test.iou5.csv', sep=',')
> bar = as.numeric(foo)
> mean(bar[1:150]); sd(bar[1:150]) # 16
[1] 0.09523626
[1] 0.06948682
> mean(bar[150:300]); sd(bar[150:300]) # 17
[1] 0.08012997
[1] 0.05101127
> mean(bar[300:450]); sd(bar[300:450]) # 18
[1] 0.06954831
[1] 0.0750822
> mean(bar[450:600]); sd(bar[450:600]) # 19
[1] 0.0243232
[1] 0.03190291
> mean(bar[600:750]); sd(bar[600:750]) # 20
[1] 0.07600799
[1] 0.05220716




> foo=read.table('~/Desktop/analysis/train.all.net7.csv', sep=';', header=TRUE)
> summary(foo)
              Path       Precision                       Recall   
 x1/frame100.png:  1   Min.   :  0.000    None              : 18  
 x1/frame101.png:  1   1st Qu.:  7.537    10.15293042040921 :  1  
 x1/frame102.png:  1   Median : 12.463    10.374422125281423:  1  
 x1/frame103.png:  1   Mean   : 25.267    100.4760194857369 :  1  
 x1/frame104.png:  1   3rd Qu.: 24.475    100.58912644926343:  1  
 x1/frame105.png:  1   Max.   :535.523    100.86077484138137:  1  
 (Other)        :694                     (Other)            :677  
      SDE              BB.IOU           IOU.20           IOU.15      
 Min.   :   0.00   Min.   :0.0000   Min.   :0.0000   Min.   :0.0000  
 1st Qu.:  52.89   1st Qu.:0.1177   1st Qu.:0.1533   1st Qu.:0.1178  
 Median : 114.18   Median :0.3336   Median :0.2647   Median :0.2242  
 Mean   : 282.04   Mean   :0.3422   Mean   :0.2495   Mean   :0.2147  
 3rd Qu.: 257.58   3rd Qu.:0.5105   3rd Qu.:0.3476   3rd Qu.:0.3052  
 Max.   :5791.40   Max.   :0.8955   Max.   :0.5419   Max.   :0.5060  
                                                                     
     IOU.10            IOU.5             IOU.0         
 Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
 1st Qu.:0.07682   1st Qu.:0.03560   1st Qu.:0.001305  
 Median :0.15810   Median :0.07712   Median :0.003810  
 Mean   :0.16164   Mean   :0.08675   Mean   :0.005950  
 3rd Qu.:0.23497   3rd Qu.:0.12390   3rd Qu.:0.008343  
 Max.   :0.45679   Max.   :0.32209   Max.   :0.061638  


> foo=read.table('~/Desktop/analysis/test.all.net7.csv', sep=';', header=TRUE)
> summary(foo)
               Path       Precision                      Recall   
 x16/frame000.png:  1   Min.   :  0.00    None              : 37  
 x16/frame001.png:  1   1st Qu.: 11.77    10.25359704952152 :  1  
 x16/frame002.png:  1   Median : 28.71    10.339111601383001:  1  
 x16/frame003.png:  1   Mean   : 44.43    10.40246982107263 :  1  
 x16/frame004.png:  1   3rd Qu.: 57.00    100.28686150188189:  1  
 x16/frame005.png:  1   Max.   :643.19    100.68899708220464:  1  
 (Other)         :744                    (Other)            :708  
      SDE              BB.IOU           IOU.20           IOU.15       
 Min.   :   0.00   Min.   :0.0000   Min.   :0.0000   Min.   :0.00000  
 1st Qu.:  65.34   1st Qu.:0.1778   1st Qu.:0.1186   1st Qu.:0.08906  
 Median : 118.58   Median :0.3317   Median :0.2191   Median :0.18344  
 Mean   : 292.40   Mean   :0.3199   Mean   :0.2029   Mean   :0.17139  
 3rd Qu.: 280.54   3rd Qu.:0.4566   3rd Qu.:0.2899   3rd Qu.:0.24617  
 Max.   :9582.98   Max.   :0.9131   Max.   :0.4905   Max.   :0.44683  
                                                                      
     IOU.10            IOU.5             IOU.0          
 Min.   :0.00000   Min.   :0.00000   Min.   :0.0000000  
 1st Qu.:0.04903   1st Qu.:0.01739   1st Qu.:0.0006139  
 Median :0.12127   Median :0.05669   Median :0.0022341  
 Mean   :0.12549   Mean   :0.06709   Mean   :0.0047498  
 3rd Qu.:0.18823   3rd Qu.:0.10184   3rd Qu.:0.0059574  
 Max.   :0.39366   Max.   :0.32501   Max.   :0.0684182  
                                                        
> sd(foo$BB.IOU)
[1] 0.2020998

