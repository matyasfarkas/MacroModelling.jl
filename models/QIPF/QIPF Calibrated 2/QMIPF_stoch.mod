// This is the (non-linear) Quantitative Microfounded Integrated Policy Framework Model
//
// By Tobias Adrian, Christopher Erceg, Marcin Kolasa, Jesper Linde, and Pawel Zabczyk
//
// This version 08/21/2022

//-------------------------------------------------------------------------------------------------------------------------)
// 1. Variable declaration
//-------------------------------------------------------------------------------------------------------------------------)

//-------------------------------------------------------------------------------------------------------------------------)
//Note: model variables are in per capita levels
//-------------------------------------------------------------------------------------------------------------------------)
var 
//-------------------------------------------------------------------------------------------------------------------------)
//NAME             $      LaTeX / Sci Word           $                               DESCRIPTION                           )
//-------------------------------------------------------------------------------------------------------------------------)
B                  $  {B}                            $  (long_name='  1. Net foreign assets (home)                        ')   
B_F                $  {B_{F}}                        $  (long_name='  2. Intermediated funds                              ')  
B_F_POT            $  {B_{F}^{pot}}                  $  (long_name='  3. Potential intermediated funds                    ') 
B_M                $  {B_{M}}                        $  (long_name='  4. FXI (home)                                       ') 
B_P                $  {B_{P}}                        $  (long_name='  5. Portfolio inflow (home)                          ') 
B_POT              $  {B^{pot}}                      $  (long_name='  6. Potential net foreign assets (home)              ') 
BLIM               $  {BLIM}                         $  (long_name='  7. Debt limit (home)                                ')  
C                  $  {C}                            $  (long_name='  8. Private consumption (home)                       ')  
C_POT              $  {C^{pot}}                      $  (long_name='  9. Potential private consumption (home)             ') 
C_ST               $  {C^{\ast}}                     $  (long_name=' 10. Private consumption (foreign)                    ')   
C_ST_POT           $  {C^{\ast,pot}}                 $  (long_name=' 11. Potential private consumption (foreign)          ') 
C_TIL              $  {\widetilde{C}}                $  (long_name=' 12. Total consumption (home)                         ')   
C_TIL_POT          $  {\widetilde{C^{pot}}}          $  (long_name=' 13. Potential total consumption (home)               ')  
C_TIL_ST           $  {\widetilde{C^{\ast}}}         $  (long_name=' 14. Total consumption (foreign)                      ')   
C_TIL_ST_POT       $  {\widetilde{C^{\ast,pot}}}     $  (long_name=' 15. Potential total consumption (foreign)            ')  
D4Q_CAL            $  {\Delta_{4}\mathcal{Q}^{cal}}  $  (long_name=' 16. Calibrated OYA real exchange rate growth (home)  ')
D4W_CAL            $  {\Delta_{4}W^{cal}}            $  (long_name=' 17. Calibrated OYA wage growth (home)                ')
D4W_CAL_ST         $  {\Delta_{4}W^{\ast,cal}}       $  (long_name=' 18. Calibrated OYA wage growth (foreign)             ')
D4Y_CAL            $  {\Delta_{4}Y^{cal}}            $  (long_name=' 19. Calibrated OYA output growth (home)              ')
D4Y_CAL_ST         $  {\Delta_{4}Y^{\ast,cal}}       $  (long_name=' 20. Calibrated OYA output growth (foreign)           ')
E_I                $  {E_{I}}                        $  (long_name=' 21. Monetary policy disturbance (home)               ')  
E_I_ST             $  {E_{I}^{\ast}}                 $  (long_name=' 22. Monetary policy disturbance (foreign)            ')  
G                  $  {G}                            $  (long_name=' 23. Government consumption (home)                    ')  
G_ST               $  {G^{\ast}}                     $  (long_name=' 24. Government consumption (foreign)                 ')   
GAM_CD             $  {\Gamma_{CD}}                  $  (long_name=' 25. Ratio of cons. to production prices (home)       ')   
GAM_CD_POT         $  {\Gamma_{CD}^{pot}}            $  (long_name=' 26. Ratio of pot. cons. to prod. prices (home)       ')  
GAM_CD_ST          $  {\Gamma_{CD}^{\ast}}           $  (long_name=' 27. Ratio of cons. to production prices (foreign)    ')   
GAM_CD_ST_POT      $  {\Gamma_{CD}^{\ast,pot}}       $  (long_name=' 28. Ratio of pot. cons. to prod. prices (foreign)    ')  
GAM_CM             $  {\Gamma_{CM}}                  $  (long_name=' 29. Ratio of cons. to import prices (home)           ')   
GAM_CM_POT         $  {\Gamma_{CM}^{pot}}            $  (long_name=' 30. Ratio of pot. cons. to import prices (home)      ')  
GAM_CM_ST          $  {\Gamma_{CM}^{\ast}}           $  (long_name=' 31. Ratio of cons. to import prices (foreign)        ')   
GAM_CM_ST_POT      $  {\Gamma_{CM}^{\ast,pot}}       $  (long_name=' 32. Ratio of pot. cons. to imp. prices (foreign)     ')  
GAM_GD             $  {\Gamma_{GD}}                  $  (long_name=' 33. Ratio of govt. cons. to prod. prices (home)      ')   
GAM_GD_POT         $  {\Gamma_{GD}^{pot}}            $  (long_name=' 34. Ratio of pot. govt. cons. to prod. prices (home) ')  
GAM_GD_ST          $  {\Gamma_{GD}^{\ast}}           $  (long_name=' 35. Ratio of govt. cons. to prod. prices (foreign)   ')   
GAM_GD_ST_POT      $  {\Gamma_{GD}^{\ast,pot}}       $  (long_name=' 36. Ratio of pot. govt. cons. to prod. prices (for.) ')  
GAM_GM             $  {\Gamma_{GM}}                  $  (long_name=' 37. Ratio of govt. cons. to import prices (home)     ')   
GAM_GM_POT         $  {\Gamma_{GM}^{pot}}            $  (long_name=' 38. Ratio of potential govt. to import prices (home) ')  
GAM_GM_ST          $  {\Gamma_{GM}^{\ast}}           $  (long_name=' 39. Ratio of govt. cons. to import prices (foreign)  ') 
GAM_GM_ST_POT      $  {\Gamma_{GM}^{\ast,pot}}       $  (long_name=' 40. Ratio of pot. govt. cons. to imp. prices (for.)  ')  
GAM_MD             $  {\Gamma_{MD}}                  $  (long_name=' 41. Ratio of import to production prices             ') 
GAM_MD_POT         $  {\Gamma_{MD}^{pot}}            $  (long_name=' 42. Ratio of potential import to production prices   ')  
GAM_MD_ST          $  {\Gamma_{MD}^{\ast}}           $  (long_name=' 43. Ratio of import to production prices (foreign)   ') 
GAMMA              $  {\Gamma}                       $  (long_name=' 44. Gabaix-Maggiori friction                         ') 
GAMMA_POT          $  {\Gamma^{pot}}                 $  (long_name=' 45. Potential Gabaix-Maggiori friction               ')  
I                  $  {I}                            $  (long_name=' 46. Nominal interest rate (home)                     ')   
I_CAL              $  {I^{cal}}                      $  (long_name=' 47. Calibrated nominal interest rate (home)          ')
I_CAL_ST           $  {I^{\ast,cal}}                 $  (long_name=' 48. Calibrated nominal interest rate (foreign)       ')
I_POT              $  {I^{pot}}                      $  (long_name=' 49. Potential nominal interest rate (home)           ')  
I_ST               $  {I^{\ast}}                     $  (long_name=' 50. Nominal interest rate (foreign)                  ')   
I_ST_POT           $  {I^{\ast,pot}}                 $  (long_name=' 51. Potential nominal interest rate (foreign)        ')  
IB                 $  {I^{B}}                        $  (long_name=' 52. Nominal retail interest rate (home)              ')   
LAM                $  {\Lambda}                      $  (long_name=' 53. Marginal utility of consumption (home)           ')   
LAM_POT            $  {\Lambda^{pot}}                $  (long_name=' 54. Potential marginal utility of consumption (home) ')  
LAM_ST             $  {\Lambda^{\ast}}               $  (long_name=' 55. Marginal utility of consumption (foreign)        ')   
LAM_ST_POT         $  {\Lambda^{\ast,pot}}           $  (long_name=' 56. Pot. marginal utility of consumption (foreign)   ')  
M_C                $  {M_{C}}                        $  (long_name=' 57. Imported component of consumption (home)         ')   
M_C_POT            $  {M_{C}^{pot}}                  $  (long_name=' 58. Pot. imported component of consumption (home)    ')  
M_C_ST             $  {M_{C}^{\ast}}                 $  (long_name=' 59. Imported component of consumption (foreign)      ')   
M_C_ST_POT         $  {M_{C}^{\ast,pot}}             $  (long_name=' 60. Pot. imported component of consumption (foreign) ')  
M_G                $  {M_{G}}                        $  (long_name=' 61. Imported component of government cons.  (home)   ')   
M_G_POT            $  {M_{G}^{pot}}                  $  (long_name=' 62. Pot. imported component of govt. cons.  (home)   ')  
M_G_ST             $  {M_{G}^{\ast}}                 $  (long_name=' 63. Imported component of government cons.  (foreign)')   
M_G_ST_POT         $  {M_{G}^{\ast,pot}}             $  (long_name=' 64. Pot. imported component of govt. cons. (foreign) ')  
MC_D               $  {MC_{D}}                       $  (long_name=' 65. Real (producer price) marginal cost (home)       ')   
MC_D_ST            $  {MC_{D}^{\ast}}                $  (long_name=' 66. Real (producer price) marginal cost (foreign)    ')   
N                  $  {N}                            $  (long_name=' 67. Labor (home)                                     ')   
N_POT              $  {N^{pot}}                      $  (long_name=' 68. Potential labor (home)                           ')  
N_ST               $  {N^{\ast}}                     $  (long_name=' 69. Labor (foreign)                                  ')   
N_ST_POT           $  {N^{\ast,pot}}                 $  (long_name=' 70. Potential labor (foreign)                        ')  
NU                 $  {\nu}                          $  (long_name=' 71. Demand indicator (home)                          ')  
NU_ST              $  {\nu^{\ast}}                   $  (long_name=' 72. Demand indicator (foreign)                       ')  
P_AMP_D            $  {P_{D}^{\#}}                   $  (long_name=' 73. Price dispersion (home)                          ')   
P_AMP_D_ST         $  {P_{D}^{\ast,\#}}              $  (long_name=' 74. Price dispersion (foreign)                       ')   
P_AMP_M            $  {P_{M}^{\#}}                   $  (long_name=' 75. Import price dispersion (home)                   ')   
P_AMP_M_ST         $  {P_{M}^{\ast,\#}}              $  (long_name=' 76. Import price dispersion (foreign)                ')   
P_TIL_D            $  {\widetilde{P_{D}}}            $  (long_name=' 77. Ratio of optimal reset price to PPI (home)       ')   
P_TIL_D_ST         $  {\widetilde{P_{D}^{\ast}}}     $  (long_name=' 78. Ratio of optimal reset price to PPI (foreign)    ')   
P_TIL_M            $  {\widetilde{P_{M}}}            $  (long_name=' 79. Optimal import reset price to imp. price (home)  ')   
P_TIL_M_ST         $  {\widetilde{P_{M}^{\ast}}}     $  (long_name=' 80. Opt. import reset price to imp. price (foreign)  ')   
PI_C               $  {\Pi_{C}}                      $  (long_name=' 81. CPI inflation (home)                             ')   
PI_C_POT           $  {\Pi_{C}^{pot}}                $  (long_name=' 82. Potential CPI inflation (home)                   ')  
PI_C_ST            $  {\Pi_{C}^{\ast}}               $  (long_name=' 83. CPI inflation (foreign)                          ')   
PI_C_ST_POT        $  {\Pi_{C}^{\ast,pot}}           $  (long_name=' 84. Potential CPI inflation (foreign)                ')  
PI_CAL             $  {\Pi^{cal}}                    $  (long_name=' 85. Calibrated inflation (home)                      ')
PI_CAL_ST          $  {\Pi^{\ast,cal}}               $  (long_name=' 86. Calibrated inflation (foreign)                   ')
PI_D               $  {\Pi_{D}}                      $  (long_name=' 87. PPI inflation (home)                             ')   
PI_D_POT           $  {\Pi_{D}^{pot}}                $  (long_name=' 88. Potential PPI inflation (home)                   ')  
PI_D_ST            $  {\Pi_{D}^{\ast}}               $  (long_name=' 89. PPI inflation (foreign)                          ')   
PI_D_ST_POT        $  {\Pi_{D}^{\ast,pot}}           $  (long_name=' 90. Potential PPI inflation (foreign)                ')  
PI_M               $  {\Pi_{M}}                      $  (long_name=' 91. Import price inflation (home)                    ')   
PI_M_ST            $  {\Pi_{M}^{\ast}}               $  (long_name=' 92. Import price inflation (foreign)                 ')   
PI_P               $  {\Pi_{P}}                      $  (long_name=' 93. Domestic price indexation factor (home)          ')   
PI_P_ST            $  {\Pi_{P}^{\ast}}               $  (long_name=' 94. Domestic price indexation factor (foreign)       ')   
PI_PM              $  {\Pi_{PM}}                     $  (long_name=' 95. Export price indexation factor (home)            ')   
PI_PM_ST           $  {\Pi_{PM}^{\ast}}              $  (long_name=' 96. Export price indexation factor (foreign)         ')   
PI_W               $  {\Pi_{W}}                      $  (long_name=' 97. Wage indexation factor (home)                    ')   
PI_W_ST            $  {\Pi_{W}^{\ast}}               $  (long_name=' 98. Wage indexation factor (foreign)                 ')   
Q                  $  {\mathcal{Q}}                  $  (long_name=' 99. Real exchange rate                               ')   
Q_CAL              $  {\mathcal{Q}^{cal}}            $  (long_name='100. Calibrated real exchange rate                    ')
Q_POT              $  {\mathcal{Q}^{pot}}            $  (long_name='101. Potential real exchange rate                     ')  
TAU_C              $  {\tau_{C}}                     $  (long_name='102. Consumption taxes (home)                         ')
TAU_C_ST           $  {\tau_{C}^{\ast}}              $  (long_name='103. Consumption taxes (foreign)                      ')
TAU_F              $  {\tau_{F}}                     $  (long_name='104. Capital inflow taxes (home)                      ') 
TAU_N              $  {\tau_{N}}                     $  (long_name='105. Labor taxes (home)                               ') 
TAU_N_ST           $  {\tau_{N}^{\ast}}              $  (long_name='106. Labor taxes (foreign)                            ') 
TB_CAL             $  {TB^{cal}}                     $  (long_name='107. Calibrated trade balance (home)                  ')
THETA              $  {\Theta}                       $  (long_name='108. Spread (home)                                    ')  
U                  $  {U}                            $  (long_name='109. Average lifetime utility (home)                  ')  
U_ST               $  {U^{\ast}}                     $  (long_name='110. Average lifetime utility (foreign)               ')  
UIP_CAL            $  {UIP^{cal}}                    $  (long_name='111. Calibrated UIP premium (home)                    ')
UPSILON            $  {\Upsilon}                     $  (long_name='112. Domestic price markup (home)                     ') 
UPSILON_M          $  {\Upsilon_{M}}                 $  (long_name='113. Export price markup (foreign)                    ') 
UPSILON_M_ST       $  {\Upsilon_{M}^{\ast}}          $  (long_name='114. Export price markup (home)                       ') 
UPSILON_ST         $  {\Upsilon^{\ast}}              $  (long_name='115. Domestic price markup (foreign)                  ') 
UPSILON_W          $  {\Upsilon_{W}}                 $  (long_name='116. Wage markup (home)                               ') 
UPSILON_W_ST       $  {\Upsilon_{W}^{\ast}}          $  (long_name='117. Wage markup (foreign)                            ') 
VARSIGMA           $  {\varsigma}                    $  (long_name='118. Preference indicator (home)                      ') 
VARSIGMA_ST        $  {\varsigma^{\ast}}             $  (long_name='119. Preference indicator (foreign)                   ') 
VARTHETA           $  {\vartheta}                    $  (long_name='120. Domestic price dispersion aux. variable (home)   ')  
VARTHETA_M         $  {\vartheta_{M}}                $  (long_name='121. Export price dispersion aux. variable (home)     ')  
VARTHETA_M_ST      $  {\vartheta_{M}^{\ast}}         $  (long_name='122. Export price dispersion aux. variable (foreign)  ')  
VARTHETA_ST        $  {\vartheta^{\ast}}             $  (long_name='123. Domestic price dispersion aux. variable (foreign)') 
W_AMP_U            $  {W_{U}^{\#}}                   $  (long_name='124. Wage dispersion for aggregate N^{1+chi} (home)   ')  
W_AMP_U_ST         $  {W_{U}^{\ast,\#}}              $  (long_name='125. Wage dispersion for aggregate N^{1+chi} (foreign)')  
W_C                $  {W_{C}}                        $  (long_name='126. Consumption real wage (home)                     ')  
W_C_POT            $  {W_{C}^{pot}}                  $  (long_name='127. Potential consumption real wage (home)           ') 
W_C_ST             $  {W_{C}^{\ast}}                 $  (long_name='128. Consumption real wage (foreign)                  ')   
W_C_ST_POT         $  {W_{C}^{\ast,pot}}             $  (long_name='129. Potential consumption real wage (foreign)        ')  
W_CAL              $  {W^{cal}}                      $  (long_name='130. Calibrated wage rate (home)                      ')
W_CAL_ST           $  {W^{\ast,cal}}                 $  (long_name='131. Calibrated wage rate (foreign)                   ')
W_TIL_C            $  {\widetilde{WC}}               $  (long_name='132. Ratio of optimal reset wage to CPI (home)        ')   
W_TIL_C_ST         $  {\widetilde{WC^{\ast}}}        $  (long_name='133. Ratio of optimal reset wage to CPI (foreign)     ')   
Y                  $  {Y}                            $  (long_name='134. GDP (home)                                       ')   
Y_CAL              $  {Y^{cal}}                      $  (long_name='135. Calibrated output (home)                         ')
Y_CAL_ST           $  {Y^{\ast,cal}}                 $  (long_name='136. Calibrated output (foreign)                      ')
Y_D                $  {Y_{D}}                        $  (long_name='137. Domestically produced goods (home)               ')   
Y_D_POT            $  {Y_{D}^{pot}}                  $  (long_name='138. Potential domestically produced goods (home)     ')  
Y_D_ST             $  {Y_{D}^{\ast}}                 $  (long_name='139. Domestically produced goods (foreign)            ')   
Y_D_ST_POT         $  {Y_{D}^{\ast,pot}}             $  (long_name='140. Potential domestically produced goods (foreign)  ')  
Y_M                $  {Y_{M}}                        $  (long_name='141. Export goods (home)                              ')   
Y_M_POT            $  {Y_{M}^{pot}}                  $  (long_name='142. Potential export goods (foreign)                 ')  
Y_M_ST             $  {Y_{M}^{\ast}}                 $  (long_name='143. Export goods (foreign)                           ')   
Y_M_ST_POT         $  {Y_{M}^{\ast,pot}}             $  (long_name='144. Potential export goods (foreign)                 ')  
Y_POT              $  {Y^{pot}}                      $  (long_name='145. Potential GDP (home)                             ')  
Y_ST               $  {Y^{\ast}}                     $  (long_name='146. GDP (foreign)                                    ')   
Y_ST_POT           $  {Y^{\ast,pot}}                 $  (long_name='147. Potential GDP (foreign)                          ')  
Z                  $  {Z}                            $  (long_name='148. Aggregate productivity (home)                    ')  
Z_1                $  {Z_{1}}                        $  (long_name='149. Auxiliary variable 1 (home)                      ')   
Z_1_ST             $  {Z_{1}^{\ast}}                 $  (long_name='150. Auxiliary variable 1 (foreign)                   ')   
Z_2                $  {Z_{2}}                        $  (long_name='151. Auxiliary variable 2 (home)                      ')   
Z_2_ST             $  {Z_{2}^{\ast}}                 $  (long_name='152. Auxiliary variable 2 (foreign)                   ')   
Z_3                $  {Z_{3}}                        $  (long_name='153. Auxiliary variable 3 (home)                      ')   
Z_3_ST             $  {Z_{3}^{\ast}}                 $  (long_name='154. Auxiliary variable 3 (foreign)                   ')   
Z_4                $  {Z_{4}}                        $  (long_name='155. Auxiliary variable 4 (home)                      ')   
Z_4_ST             $  {Z_{4}^{\ast}}                 $  (long_name='156. Auxiliary variable 4 (foreign)                   ')   
Z_5                $  {Z_{5}}                        $  (long_name='157. Auxiliary variable 5 (home)                      ')   
Z_5_ST             $  {Z_{5}^{\ast}}                 $  (long_name='158. Auxiliary variable 5 (foreign)                   ')   
Z_6                $  {Z_{6}}                        $  (long_name='159. Auxiliary variable 6 (home)                      ')   
Z_6_ST             $  {Z_{6}^{\ast}}                 $  (long_name='160. Auxiliary variable 6 (foreign)                   ')   
Z_7                $  {Z_{7}}                        $  (long_name='161. Auxiliary variable 7 (home)                      ')   
Z_7_ST             $  {Z_{7}^{\ast}}                 $  (long_name='162. Auxiliary variable 7 (foreign)                   ')   
Z_8                $  {Z_{8}}                        $  (long_name='163. Auxiliary variable 8 (home)                      ')   
Z_8_ST             $  {Z_{8}^{\ast}}                 $  (long_name='164. Auxiliary variable 8 (foreign)                   ')   
Z_M_1              $  {{Z_{M}}_{1}}                  $  (long_name='165. Auxiliary export variable 1 (home)               ')   
Z_M_1_ST           $  {{Z_{M}}_{1}^{\ast}}           $  (long_name='166. Auxiliary export variable 1 (foreign)            ')   
Z_M_2              $  {{Z_{M}}_{2}}                  $  (long_name='167. Auxiliary export variable 2 (home)               ')   
Z_M_2_ST           $  {{Z_{M}}_{2}^{\ast}}           $  (long_name='168. Auxiliary export variable 2 (foreign)            ')   
Z_M_3              $  {{Z_{M}}_{3}}                  $  (long_name='169. Auxiliary export variable 3 (home)               ')   
Z_M_3_ST           $  {{Z_{M}}_{3}^{\ast}}           $  (long_name='170. Auxiliary export variable 3 (foreign)            ')   
Z_M_4              $  {{Z_{M}}_{4}}                  $  (long_name='171. Auxiliary export variable 4 (home)               ')   
Z_M_4_ST           $  {{Z_{M}}_{4}^{\ast}}           $  (long_name='172. Auxiliary export variable 4 (foreign)            ')   
Z_M_5              $  {{Z_{M}}_{5}}                  $  (long_name='173. Auxiliary export variable 5 (home)               ')   
Z_M_5_ST           $  {{Z_{M}}_{5}^{\ast}}           $  (long_name='174. Auxiliary export variable 5 (foreign)            ')   
Z_M_6              $  {{Z_{M}}_{6}}                  $  (long_name='175. Auxiliary export variable 6 (home)               ')   
Z_M_6_ST           $  {{Z_{M}}_{6}^{\ast}}           $  (long_name='176. Auxiliary export variable 6 (foreign)            ')
Z_ST               $  {Z^{\ast}}                     $  (long_name='177. Aggregate productivity (foreign)                 ')
;//------------------------------------------------------------------------------------------------------------------------)          

varexo                                                                                                    
//-------------------------------------------------------------------------------------------------------------------------)
//NAME             $      LaTeX / Sci Word           $                               DESCRIPTION                           )
//-------------------------------------------------------------------------------------------------------------------------)
EPS_B_M            $  {\varepsilon_{M}}              $  (long_name='  1. FXI shock (home)                                 ')
EPS_B_P            $  {\varepsilon_{P}}              $  (long_name='  2. Portfolio inflow shock (home)                    ')
EPS_G              $  {\varepsilon_{G}}              $  (long_name='  3. Government consumption shock (home)              ')
EPS_G_ST           $  {\varepsilon_{G}^{\ast}}       $  (long_name='  4. Government consumption shock (foreign)           ')
EPS_I              $  {\varepsilon_{I}}              $  (long_name='  5. Monetary policy shock (home)                     ')
EPS_I_ST           $  {\varepsilon_{I}^{\ast}}       $  (long_name='  6. Monetary policy shock (foreign)                  ')
EPS_NU             $  {\varepsilon_{\nu}}            $  (long_name='  7. Demand shock (home)                              ')
EPS_NU_ST          $  {\varepsilon_{\nu}^{\ast}}     $  (long_name='  8. Demand shock (foreign)                           ')
EPS_TAU_C          $  {\varepsilon_{\tau_C}}         $  (long_name='  9. Consumption tax shock (home)                     ')
EPS_TAU_C_ST       $  {\varepsilon_{\tau_C}^{\ast}}  $  (long_name=' 10. Consumption tax shock (foreign)                  ')
EPS_TAU_F          $  {\varepsilon_{\tau_F}}         $  (long_name=' 11. Capital inflow tax shock (home)                  ')
EPS_TAU_N          $  {\varepsilon_{\tau_N}}         $  (long_name=' 12. Labor tax shock (home)                           ')
EPS_TAU_N_ST       $  {\varepsilon_{\tau_N}^{\ast}}  $  (long_name=' 13. Labor tax shock (foreign)                        ')
EPS_UPSILON        $  {\varepsilon_{\Upsilon}}       $  (long_name=' 14. Domestic sales price markup shock (home)         ')
EPS_UPSILON_M      $  {\varepsilon_{\Upsilon_M}}     $  (long_name=' 15. Import sales price markup shock (foreign)        ')
EPS_UPSILON_M_ST   ${\varepsilon_{\Upsilon_M}^{\ast}}$  (long_name=' 16. Import sales price markup shock (home)           ')
EPS_UPSILON_ST     ${\varepsilon_{\Upsilon}^{\ast}}  $  (long_name=' 17. Domestic sales price markup shock (foreign)      ')
EPS_UPSILON_W      $  {\varepsilon_{\Upsilon_W}}     $  (long_name=' 18. Wage markup shock (home)                         ')
EPS_UPSILON_W_ST   ${\varepsilon_{\Upsilon_W}^{\ast}}$  (long_name=' 19. Wage markup shock (foreign)                      ')
EPS_VARSIGMA       $  {\varepsilon_{\varsigma}}      $  (long_name=' 20. Preference shock (home)                          ')
EPS_VARSIGMA_ST    ${\varepsilon_{\varsigma}^{\ast}} $  (long_name=' 21. Preference shock (foreign)                       ')
EPS_Z              $  {\varepsilon_{Z}}              $  (long_name=' 22. Aggregate productivity shock (home)              ')
EPS_Z_ST           $  {\varepsilon_{Z}^{\ast}}       $  (long_name=' 23. Aggregate productivity shock (foreign)           ')
;//------------------------------------------------------------------------------------------------------------------------)
                   
//-------------------------------------------------------------------------------------------------------------------------)
// 2. Parameter declaration
//-------------------------------------------------------------------------------------------------------------------------)
parameters
//-------------------------------------------------------------------------------------------------------------------------)
//                                                   CORE PARAMETERS                                                       )
//-------------------------------------------------------------------------------------------------------------------------)
//-------------------------------------------------------------------------------------------------------------------------)
//NAME             $     LaTeX / Sci Word            $                               DESCRIPTION                           )
//-------------------------------------------------------------------------------------------------------------------------)
alpha              $  {\alpha}                       $  (long_name='  1. Coefficient on capital in the prod. function     ')
b_y                $  {B/Y}                          $  (long_name='  2. Debt to GDP ratio                                ')
beta               $  {\beta}                        $  (long_name='  3. Deterministic, quarterly discount factor (home)  ')
beta_ST            $  {\beta^{*}}                    $  (long_name='  4. Deterministic, quart. discount factor (foreign)  ')
c_y                $  {C/Y}                          $  (long_name='  5. Consumption to GDP ratio                         ')
cfm_nonfa          $  {cfm\_nonfa}                   $  (long_name='  6. Switch moving from price to quantity CFMs        ')
chi                $  {\chi}                         $  (long_name='  7. Inverse of Frisch elasticity of labor Supply     ')
chi_0              $  {\chi_{0}}                     $  (long_name='  8. Relative weight on labor disutility (home)       ')
chi_0_ST           $  {\chi_{0}^{*}}                 $  (long_name='  9. Relative weight on labor disutility (foreign)    ')
elb                $  {elb}                          $  (long_name=' 10. Gross nominal int. rate at ELB (home)            ')
elb_ST             $  {elb^{*}}                      $  (long_name=' 11. Gross nominal int. rate at ELB (foreign)         ')
eta_0              $  {\eta_{0}}                     $  (long_name=' 12. Private utility from government consumption      ')
gamma_0            $  {\gamma_{0}}                   $  (long_name=' 13. Gamma lev. (to fit Adler-Lisack-Mano FXI effects ') 
gamma_1            $  {\gamma_{1}}                   $  (long_name=' 14. Gamma semi-elasticity wrt ER variance            ') 
iota               $  {\iota}                        $  (long_name=' 15. Degree of domestic price indexation (home)       ')
iota_e             $  {\iota_{e}}                    $  (long_name=' 16. Weight of exchange rate in wage indexation       ')
iota_m             $  {\iota_{m}}                    $  (long_name=' 17. Degree of export price indexation (home)         ')
iota_m_ST          $  {\iota_{m}^{*}}                $  (long_name=' 18. Degree of export price indexation (foreign)      ')
iota_ST            $  {\iota^{*}}                    $  (long_name=' 19. Domestic price indexation (foreign, SW 2007)     ')
iota_w             $  {\iota_{w}}                    $  (long_name=' 20. Degree of wage indexation (home)                 ')
iota_w_ST          $  {\iota_{w}^{*}}                $  (long_name=' 21. Degree of wage indexation (foreign)              ')
k                  $  {k}                            $  (long_name=' 22. Capital (home)                                   ')
k_n                $  {K/N}                          $  (long_name=' 23. Capital labor ratio (home)                       ')
k_n_ST             $  {K^{*}/N^{*}}                  $  (long_name=' 24. Capital labor ratio (foreign)                    ')
k_ST               $  {k^{*}}                        $  (long_name=' 25. Capital (foreign)                                ')
m                  $  {m}                            $  (long_name=' 26. Tightness of external debt constraint            ') 
nu                 $  {\nu}                          $  (long_name=' 27. Long memory in wage indexation term (home)       ')
nu_ST              $  {\nu^{*}}                      $  (long_name=' 28. Long memory in wage indexation term (foreign)    ')
omega_b            $  {\omega_{b}}                   $  (long_name=' 29. Domestic ownership share of banks                ') 
omega_c            $  {\omega_{c}}                   $  (long_name=' 30. Consumption share of domestic goods (in home C)  ')
omega_c_ST         $  {\omega_{c}^{\ast}}            $  (long_name=' 31. Cons. share of foreign goods (in foreign C)      ') 
omega_f            $  {\omega_{f}}                   $  (long_name=' 32. Domestic ownership share of financiers           ') 
omega_g            $  {\omega_{g}}                   $  (long_name=' 33. Govt. cons. share of domestic goods (home G)     ') 
omega_g_ST         $  {\omega_{g}^{\ast}}            $  (long_name=' 34. Govt. cons. share of foreign goods (foreign G)   ') 
omega_p            $  {\omega_{p}}                   $  (long_name=' 35. Domestic ownership share of portfolio investors  ')
psi                $  {\psi}                         $  (long_name=' 36. Weight on Kimball (domestic sales, home)         ')
psi_i              $  {\psi_{i}}                     $  (long_name=' 37. Policy rate smoothing in domestic Taylor rule    ') 
psi_i_ST           $  {\psi_{i}^{\ast}}              $  (long_name=' 38. Policy rate smoothing in foreign Taylor rule     ') 
psi_m              $  {\psi_m}                       $  (long_name=' 39. Weight on Kimball (exports, home)                ')
psi_m_ST           $  {\psi_m^{\ast}}                $  (long_name=' 40. Weight on Kimball (exports, foreign)             ')
psi_pi             $  {\psi_{\pi}}                   $  (long_name=' 41. Coeff. on PI_C in domestic Taylor rule           ') 
psi_pi_ST          $  {\psi_{\pi}^{\ast}}            $  (long_name=' 42. Coeff. on PI_C_ST in foreign Taylor rule         ') 
psi_pid            $  {\psi_{\pi_d}}                 $  (long_name=' 43. Coeff. on PI_D in domestic Taylor rule           ') 
psi_pid_ST         $  {\psi_{\pi_d}^{\ast}}          $  (long_name=' 44. Coeff. on PI_D_ST in foreign Taylor rule         ') 
psi_ST             $  {\psi^{\ast}}                  $  (long_name=' 45. Weight on Kimball (domestic sales, foreign)      ')
psi_theta          $  {\psi_{\theta}}                $  (long_name=' 46. Coeff. on spread in domestic Taylor rule         ') 
psi_x              $  {\psi_{x}}                     $  (long_name=' 47. Coeff. on output gap in domestic Taylor rule     ') 
psi_x_ST           $  {\psi_{x}^{\ast}}              $  (long_name=' 48. Coeff. on output gap in foreign Taylor rule      ') 
rho_c              $  {\rho_{c}}                     $  (long_name=' 49. Elasticity of substitution (home - foreign C)    ') 
rho_c_ST           $  {\rho_{c}^{\ast}}              $  (long_name=' 50. Elasticity of substitution (foreign - home C)    ') 
rho_e_i            $  {\rho_{e,i}}                   $  (long_name=' 51. Persistence of monetary policy shock (home)      ') 
rho_e_i_ST         $  {\rho_{e,i}^{\ast}}            $  (long_name=' 52. Persistence of monetary policy shock (foreign)   ') 
rho_g              $  {\rho_{g}}                     $  (long_name=' 53. Elasticity of substitution (home - foreign G)    ') 
rho_g_ST           $  {\rho_{g}^{\ast}}              $  (long_name=' 54. Elasticity of substitution (foreign - home G)    ') 
rho_nu             $  {\rho_{\nu}}                   $  (long_name=' 55. Persistence of demand shock (home)               ') 
rho_nu_ST          $  {\rho_{\nu}^{\ast}}            $  (long_name=' 56. Persistence of demand shock (foreign)            ') 
rho_tau_c          $  {\rho_{\tau_{c}}}              $  (long_name=' 57. Persistence of consumption taxation (home)       ') 
rho_tau_c_ST       $  {\rho_{\tau_{c}}^{\ast}}       $  (long_name=' 58. Persistence of consumption taxation (foreign)    ') 
rho_tau_f          $  {\rho_{\tau_{f}}}              $  (long_name=' 59. Persistence of capital inflow taxation (home)    ')
rho_tau_n          $  {\rho_{\tau_{n}}}              $  (long_name=' 60. Persistence of labor taxation (home)             ') 
rho_tau_n_ST       $  {\rho_{\tau_{n}}^{\ast}}       $  (long_name=' 61. Persistence of labor taxation (foreign)          ') 
rho_upsilon        $  {\rho_{\upsilon}}              $  (long_name=' 62. Persistence of domestic markup shock (home)      ') 
rho_upsilon_m      $  {\rho_{\upsilon_m}}            $  (long_name=' 63. Persistence of import markup shock (foreign)     ') 
rho_upsilon_m_ST   $  {\rho_{\upsilon_m}^{\ast}}     $  (long_name=' 64. Persistence of import markup shock (home)        ') 
rho_upsilon_ST     $  {\rho_{\upsilon}^{\ast}}       $  (long_name=' 65. Persistence of domestic markup shock (foreign)   ') 
rho_upsilon_w      $  {\rho_{\upsilon_w}}            $  (long_name=' 66. Persistence of wage markup shock (home)          ') 
rho_upsilon_w_ST   $  {\rho_{\upsilon_w}^{\ast}}     $  (long_name=' 67. Persistence of wage markup shock (foreign)       ') 
rho_varsigma       $  {\rho_{\varsigma}}             $  (long_name=' 68. Persistence of preference shock (home)           ') 
rho_varsigma_ST    $  {\rho_{\varsigma}^{\ast}}      $  (long_name=' 69. Persistence of preference shock (foreign)        ') 
rho_z              $  {\rho_{z}}                     $  (long_name=' 70. Persistence of agg. productivity shock (home)    ') 
rho_z_ST           $  {\rho_{z}^{\ast}}              $  (long_name=' 71. Persistence of aggregate prod. shock (foreign)   ') 
s_gy               $  {s_{gy}}                       $  (long_name=' 72. Share of govt. consumption in GDP (home)         ') 
s_gy_ST            $  {s_{gy}^{\ast}}                $  (long_name=' 73. Share of govt. consumption in GDP (foreign)      ') 
s_my               $  {s_{my}}                       $  (long_name=' 74. SS FX reserves to GDP                            ')
s_omega_g_c_ST     $  {s_{\omega,g,c}^{ast}}         $  (long_name=' 75. Ratio of foreign govt. to private home bias pars.')
s_py               $  {s_{py}}                       $  (long_name=' 76. SS portfolio investors position as a share of GDP') 
sigma              $  {\sigma}                       $  (long_name=' 77. Intertemporal elasticity of substitution         ') 
spill_i            $  {spill_{i}}                    $  (long_name=' 78. First parameter controlling spillovers           ')
spill_upsilon_w    $  {spill_{\upsilon_w}}           $  (long_name=' 79. Second parameter controlling spillovers          ')
spill_varsigma     $  {spill_{varsigma}}             $  (long_name=' 80. Third parameter controlling spillovers           ')
spill_z            $  {spill_{z}}                    $  (long_name=' 81. Fourth parameter controlling spillovers          ')
tau_p              $  {\tau_{p}}                     $  (long_name=' 82. Average and steady state production subsidy      ')
tau_w              $  {\tau_{w}}                     $  (long_name=' 83. Average and steady state labor subsidy           ')
theta_p            $  {\theta_{p}}                   $  (long_name=' 84. Net markup in product markets (without subsidy)  ') 
theta_w            $  {\theta_{w}}                   $  (long_name=' 85. Net markup in labor market (without subsidy)     ')
var_e              $  {var_e}                        $  (long_name=' 86. Average conditional variance of the exchange rate') 
varkappa           $  {\varkappa}                    $  (long_name=' 87. Habit parameter (home)                           ') 
varkappa_ST        $  {\varkappa^{\ast}}             $  (long_name=' 88. Habit parameter (foreign)                        ') 
varrho_g           $  {\varrho_{g}}                  $  (long_name=' 89. Persistence of government spending (home)        ') 
varrho_g_ST        $  {\varrho_{g}^{\ast}}           $  (long_name=' 90. Persistence of government spending (foreign)     ') 
varrho_m           $  {\varrho_{m}}                  $  (long_name=' 91. Persistence of FXI                               ') 
varrho_p           $  {\varrho_{p}}                  $  (long_name=' 92. Persistence of portfolio shock                   ') 
xi_m               $  {\xi_{m}}                      $  (long_name=' 93. Calvo probability for exports (home)             ')
xi_m_ST            $  {\xi_{m}^{*}}                  $  (long_name=' 94. Calvo probability for exports (foreign)          ')
xi_p               $  {\xi_{p}}                      $  (long_name=' 95. Calvo prob. for domestic sales prices (home)     ') 
xi_p_ST            $  {\xi_{p}^{\ast}}               $  (long_name=' 96. Calvo prob. for domestic sales prices (foreign)  ') 
xi_w               $  {\xi_{w}}                      $  (long_name=' 97. Calvo prob. for wages (home)                     ')
xi_w_ST            $  {\xi_{w}^{\ast}}               $  (long_name=' 98. Calvo prob. for wages (foreign, as in SW 2007)   ')
z7_corr            $  {z7\_corr}                     $  (long_name=' 99. Z7 correction factor (home)                      ')
z7_corr_ST         $  {z7\_corr^{\ast}}              $  (long_name='100. Z7 correction factor (foreign)                   ')
z8_corr            $  {z8\_corr}                     $  (long_name='101. Z8 correction factor (home)                      ')
z8_corr_ST         $  {z8\_corr^{\ast}}              $  (long_name='102. Z8 correction factor (foreign)                   ')
zeta               $  {\zeta}                        $  (long_name='103. Relative size of home economy                    ')
zeta_ST            $  {\zeta^{\ast}}                 $  (long_name='104. Relative size of foreign economy                 ')
m_by //distance from debt limit (in percent of GDP)
sim_mode
//-------------------------------------------------------------------------------------------------------------------------) 
//                                             STEADY STATE PARAMETERS                                                     ) 
//-------------------------------------------------------------------------------------------------------------------------)                    
SS_B               $  {\overbar{B}}                  $  (long_name='  S1. SS net foreign assets (home)                    ')
SS_B_F             $  {\overbar{B_{F}}}              $  (long_name='  S2. SS intermediated funds                          ')
SS_B_M             $  {\overbar{B_{M}}}              $  (long_name='  S3. SS FXI (home)                                   ')
SS_B_P             $  {\overbar{B_{P}}}              $  (long_name='  S4. SS portfolio inflow (home)                      ')
SS_BLIM            $  {\overbar{BLIM}}               $  (long_name='  S5. SS debt limit (home, auxiliary variable)        ')
SS_C               $  {\overbar{C}}                  $  (long_name='  S6. SS private consumption (home)                   ')
SS_C_ST            $  {\overbar{C^{\ast}}}           $  (long_name='  S7. SS private consumption (foreign)                ')
SS_C_TIL           $  {\overbar{\widetilde{C}}}      $  (long_name='  S8. SS total consumption (home)                     ')
SS_C_TIL_ST        $  {\overbar{\widetilde{C^{\ast}}}}$ (long_name='  S9. SS total consumption (foreign)                  ')
SS_E_I             $  {\overbar{E_{I}}}              $  (long_name=' S10. SS monetary policy disturbance (home)           ')
SS_E_I_ST          $  {\overbar{E_{I}^{\ast}}}       $  (long_name=' S11. SS monetary policy disturbance (foreign)        ')
SS_G               $  {\overbar{G}}                  $  (long_name=' S12. SS government consumption (home)                ')
SS_G_ST            $  {\overbar{G^{\ast}}}           $  (long_name=' S13. SS government consumption (foreign)             ')
SS_GAM_CD          $  {\overbar{\Gamma_{CD}}}        $  (long_name=' S14. SS ratio of cons. to production prices (home)   ')
SS_GAM_CD_ST       $  {\overbar{\Gamma_{CD}^{\ast}}} $  (long_name=' S15. SS ratio of cons. to production prices (foreign)')
SS_GAM_CM          $  {\overbar{\Gamma_{CM}}}        $  (long_name=' S16. SS ratio of cons. to import prices (home)       ')
SS_GAM_CM_ST       $  {\overbar{\Gamma_{CM}^{\ast}}} $  (long_name=' S17. SS ratio of cons. to import prices (foreign)    ')
SS_GAM_GD          $  {\overbar{\Gamma_{GD}}}        $  (long_name=' S18. SS ratio of govt. cons. to prod. prices (home)  ')
SS_GAM_GD_ST       $  {\overbar{\Gamma_{GD}^{\ast}}} $  (long_name=' S19. SS ratio of gov. cons. to prod. prices (foreign)')
SS_GAM_GM          $  {\overbar{\Gamma_{GM}}}        $  (long_name=' S20. SS ratio of govt. cons to import prices (home)  ')
SS_GAM_GM_ST       $  {\overbar{\Gamma_{GM}^{\ast}}} $  (long_name=' S21. SS ratio of govt. cons. to imp. prices (foreign)')
SS_GAM_MD          $  {\overbar{\Gamma_{MD}}}        $  (long_name=' S22. SS ratio of import to production prices         ')
SS_GAM_MD_ST       $  {\overbar{\Gamma_{MD}^{\ast}}} $  (long_name=' S23. SS ratio of import to prod. prices (foreign)    ')
SS_GAMMA           $  {\overbar{\Gamma}}             $  (long_name=' S24. SS Gabaix-Maggiori friction                     ')
SS_I               $  {\overbar{I}}                  $  (long_name=' S25. SS nominal interest rate (home)                 ')
SS_I_ST            $  {\overbar{I^{\ast}}}           $  (long_name=' S26. SS nominal interest rate (foreign)              ')
SS_IB              $  {\overbar{I^{B}}}              $  (long_name=' S27. SS nominal retail interest rate (home)          ')
SS_LAM             $  {\overbar{\Lambda}}            $  (long_name=' S28. SS marginal utility of consumption (home)       ')
SS_LAM_ST          $  {\overbar{\Lambda^{\ast}}}     $  (long_name=' S29. SS marginal utility of consumption (foreign)    ')
SS_M_C             $  {\overbar{M_{C}}}              $  (long_name=' S30. SS imported component of consumption (home)     ')
SS_M_C_ST          $  {\overbar{M_{C}^{\ast}}}       $  (long_name=' S31. SS imported component of consumption (foreign)  ')
SS_M_G             $  {\overbar{M_{G}}}              $  (long_name=' S32. SS imported component of govt. cons. (home)     ')
SS_M_G_ST          $  {\overbar{M_{G}^{\ast}}}       $  (long_name=' S33. SS imported component of govt. cons. (foreign)  ')
SS_MC_D            $  {\overbar{MC_{D}}}             $  (long_name=' S34. SS real (producer price) marginal cost (home)   ')
SS_MC_D_ST         $  {\overbar{MC_{D}^{\ast}}}      $  (long_name=' S35. SS real (producer price) marginal cost (foreign)')
SS_N               $  {\overbar{N}}                  $  (long_name=' S36. SS labor (home)                                 ')
SS_N_ST            $  {\overbar{N^{\ast}}}           $  (long_name=' S37. SS labor (foreign)                              ')
SS_NU              $  {\overbar{\nu}}                $  (long_name=' S38. SS demand indicator (home)                      ')
SS_NU_ST           $  {\overbar{\nu^{\ast}}}         $  (long_name=' S39. SS demand indicator (foreign)                   ')
SS_P_AMP_D         $  {\overbar{P_{D}^{\#}}}         $  (long_name=' S40. SS price dispersion (home)                      ')
SS_P_AMP_D_ST      $  {\overbar{P_{D}^{\ast ,\#}}}   $  (long_name=' S41. SS price dispersion (foreign)                   ')
SS_P_AMP_M         $  {\overbar{P_{M}^{\#}}}         $  (long_name=' S42. SS import price dispersion (home)               ')
SS_P_AMP_M_ST      $  {\overbar{P_{M}^{\ast ,\#}}}   $  (long_name=' S43. SS import price dispersion (foreign)            ')
SS_P_TIL_D         $  {\overbar{\widetilde{P_{D}}}}  $  (long_name=' S44. SS ratio of optimal reset price to PPI (home)   ')
SS_P_TIL_D_ST $  {\overbar{\widetilde{P_{D}^{\ast}}}}$  (long_name=' S45. SS ratio of opt. reset price to PPI (foreign)   ')
SS_P_TIL_M         $  {\overbar{\widetilde{P_{M}}}}  $  (long_name=' S46. SS optimal imp. reset price to imp. price (home)')
SS_P_TIL_M_ST   ${\overbar{\widetilde{P_{M}^{\ast}}}}$  (long_name=' S47. SS opt. imp. reset price to imp. price (foreign)')
SS_PI_C            $  {\overbar{\Pi_{C}}}            $  (long_name=' S48. SS CPI inflation (home)                         ')
SS_PI_C_ST         $  {\overbar{\Pi_{C}^{\ast}}}     $  (long_name=' S49. SS CPI inflation (foreign)                      ')
SS_PI_D            $  {\overbar{\Pi_{D}}}            $  (long_name=' S50. SS PPI inflation (home)                         ')
SS_PI_D_ST         $  {\overbar{\Pi_{D}^{\ast}}}     $  (long_name=' S51. SS PPI inflation (foreign)                      ')
SS_PI_M            $  {\overbar{\Pi_{M}}}            $  (long_name=' S52. SS import price inflation (home)                ')
SS_PI_M_ST         $  {\overbar{\Pi_{M}^{\ast}}}     $  (long_name=' S53. SS import price inflation (foreign)             ')
SS_PI_P            $  {\overbar{\Pi_{P}}}            $  (long_name=' S54. SS domestic price indexation factor (home)      ')
SS_PI_P_ST         $  {\overbar{\Pi_{P}^{\ast}}}     $  (long_name=' S55. SS domestic price indexation factor (foreign)   ')
SS_PI_PM           $  {\overbar{\Pi_{PM}}}           $  (long_name=' S56. SS export price indexation factor (foreign)     ')
SS_PI_PM_ST        $  {\overbar{\Pi_{PM}^{\ast}}}    $  (long_name=' S57. SS export price indexation factor (home)        ')
SS_PI_W            $  {\overbar{\Pi_{W}}}            $  (long_name=' S58. SS wage indexation factor (home)                ')
SS_PI_W_ST         $  {\overbar{\Pi_{W}^{\ast}}}     $  (long_name=' S59. SS wage indexation factor (foreign)             ')
SS_Q               $  {\overbar{\mathcal{Q}}}        $  (long_name=' S60. SS real exchange rate                           ')
SS_TAU_C           $  {\overbar{\tau_{C}}}           $  (long_name=' S61. SS consumption taxes (home)                     ')
SS_TAU_C_ST        $  {\overbar{\tau_{C}^{\ast}}}    $  (long_name=' S62. SS consumption taxes (foreign)                  ')
SS_TAU_F           $  {\overbar{\tau_{F}}}           $  (long_name=' S63. SS capital inflow taxes (home)                  ')
SS_TAU_N           $  {\overbar{\tau_{N}}}           $  (long_name=' S64. SS labor taxes (home)                           ')
SS_TAU_N_ST        $  {\overbar{\tau_{N}^{\ast}}}    $  (long_name=' S65. SS labor taxes (foreign)                        ')
SS_THETA           $  {\overbar{\Theta}}             $  (long_name=' S66. SS spread (home)                                ')
SS_U               $  {\overbar{U}}                  $  (long_name=' S67. SS average lifetime utility (home)              ')
SS_U_ST            $  {\overbar{U^{\ast}}}           $  (long_name=' S68. SS average lifetime utility (foreign)           ')
SS_UPSILON         $  {\overbar{\Upsilon}}           $  (long_name=' S69. SS domestic price markup (home)                 ')
SS_UPSILON_M       $  {\overbar{\Upsilon_{M}}}       $  (long_name=' S70. SS export price markup (foreign)                ')
SS_UPSILON_M_ST    $  {\overbar{\Upsilon_{M}^{\ast}}}$  (long_name=' S71. SS export price markup (home)                   ')
SS_UPSILON_ST      $  {\overbar{\Upsilon^{\ast}}}    $  (long_name=' S72. SS price markup on domestic sales (foreign)     ')
SS_UPSILON_W       $  {\overbar{\Upsilon_{W}}}       $  (long_name=' S73. SS wage markup (home)                           ')
SS_UPSILON_W_ST    $  {\overbar{\Upsilon_{W}^{\ast}}}$  (long_name=' S74. SS wage markup (foreign)                        ')
SS_VARSIGMA        $  {\overbar{\varsigma}}          $  (long_name=' S75. SS preference indicator (home)                  ')
SS_VARSIGMA_ST     $  {\overbar{\varsigma^{\ast}}}   $  (long_name=' S76. SS preference indicator (foreign)               ')
SS_VARTHETA        $  {\overbar{\vartheta}}          $  (long_name=' S77. SS domestic price disp. aux. variable (home)    ')
SS_VARTHETA_M      $  {\overbar{\vartheta_M}}        $  (long_name=' S78. SS export price dispersion aux. variable (home) ')
SS_VARTHETA_M_ST   $  {\overbar{\vartheta_M^{\ast}}} $  (long_name=' S79. SS export price disp. aux. variable (foreign)   ')
SS_VARTHETA_ST     $  {\overbar{\vartheta^{\ast}}}   $  (long_name=' S80. SS domestic price disp. aux. variable (foreign) ')
SS_W_AMP_U         $  {\overbar{W_{U}^{\#}}}         $  (long_name=' S81. SS wage dispersion for agg. N^{1+chi} (home)    ')
SS_W_AMP_U_ST      $  {\overbar{W_{U}^{\ast, \#}}}   $  (long_name=' S82. SS wage disp. for aggregate N^{1+chi} (foreign) ')
SS_W_C             $  {\overbar{W_{C}}}              $  (long_name=' S83. SS consumption real wage (home)                 ')
SS_W_C_ST          $  {\overbar{W_{C}^{\ast}}}       $  (long_name=' S84. SS consumption real wage (foreign)              ')
SS_W_TIL_C         $  {\overbar{\widetilde{W}_{C}}}  $  (long_name=' S85. SS ratio of optimal reset wage to CPI (home)    ')
SS_W_TIL_C_ST   ${\overbar{\widetilde{W}_{C}^{\ast}}}$  (long_name=' S86. SS ratio of opt. reset wage to CPI (foreign)    ')
SS_Y               $  {\overbar{Y}}                  $  (long_name=' S87. SS GDP (home)                                   ')
SS_Y_D             $  {\overbar{Y_{D}}}              $  (long_name=' S88. SS domestically produced goods (home)           ')
SS_Y_D_ST          $  {\overbar{Y_{D}^{\ast}}}       $  (long_name=' S89. SS domestically produced goods (foreign)        ')
SS_Y_M             $  {\overbar{Y_{M}}}              $  (long_name=' S90. SS export goods (home)                          ')
SS_Y_M_ST          $  {\overbar{Y_{M}^{\ast}}}       $  (long_name=' S91. SS export goods (foreign)                       ')
SS_Y_ST            $  {\overbar{Y^{\ast}}}           $  (long_name=' S92. SS GDP (foreign)                                ')
SS_Z               $  {\overbar{Z}}                  $  (long_name=' S93. SS aggregate productivity (home)                ')
SS_Z_1             $  {\overbar{Z_{1}}}              $  (long_name=' S94. SS auxiliary variable 1 (home)                  ')
SS_Z_1_ST          $  {\overbar{Z_{1}^{\ast}}}       $  (long_name=' S95. SS auxiliary variable 1 (foreign)               ')
SS_Z_2             $  {\overbar{Z_{2}}}              $  (long_name=' S96. SS auxiliary variable 2 (home)                  ')
SS_Z_2_ST          $  {\overbar{Z_{2}^{\ast}}}       $  (long_name=' S97. SS auxiliary variable 2 (foreign)               ')
SS_Z_3             $  {\overbar{Z_{3}}}              $  (long_name=' S98. SS auxiliary variable 3 (home)                  ')
SS_Z_3_ST          $  {\overbar{Z_{3}^{\ast}}}       $  (long_name=' S99. SS auxiliary variable 3 (foreign)               ')
SS_Z_4             $  {\overbar{Z_{4}}}              $  (long_name='S100. SS auxiliary variable 4 (home)                  ')
SS_Z_4_ST          $  {\overbar{Z_{4}^{\ast}}}       $  (long_name='S101. SS auxiliary variable 4 (foreign)               ')
SS_Z_5             $  {\overbar{Z_{5}}}              $  (long_name='S102. SS auxiliary variable 5 (home)                  ')
SS_Z_5_ST          $  {\overbar{Z_{5}^{\ast}}}       $  (long_name='S103. SS auxiliary variable 5 (foreign)               ')
SS_Z_6             $  {\overbar{Z_{6}}}              $  (long_name='S104. SS auxiliary variable 6 (home)                  ')
SS_Z_6_ST          $  {\overbar{Z_{6}^{\ast}}}       $  (long_name='S105. SS auxiliary variable 6 (foreign)               ')
SS_Z_7             $  {\overbar{Z_{7}}}              $  (long_name='S106. SS auxiliary variable 7 (home)                  ')
SS_Z_7_ST          $  {\overbar{Z_{7}^{\ast}}}       $  (long_name='S107. SS auxiliary variable 7 (foreign)               ')
SS_Z_8             $  {\overbar{Z_{8}}}              $  (long_name='S108. SS auxiliary variable 8 (home)                  ')
SS_Z_8_ST          $  {\overbar{Z_{8}^{\ast}}}       $  (long_name='S109. SS auxiliary variable 8 (foreign)               ')
SS_Z_M_1           $  {\overbar{{Z_{M}}_{1}}}        $  (long_name='S110. SS auxiliary export variable 1 (home)           ')
SS_Z_M_1_ST        $  {\overbar{{Z_{M}}_{1}^{\ast}}} $  (long_name='S111. SS auxiliary export variable 1 (foreign)        ')
SS_Z_M_2           $  {\overbar{{Z_{M}}_{2}}}        $  (long_name='S112. SS auxiliary export variable 2 (home)           ')
SS_Z_M_2_ST        $  {\overbar{{Z_{M}}_{2}^{\ast}}} $  (long_name='S113. SS auxiliary export variable 2 (foreign)        ')
SS_Z_M_3           $  {\overbar{{Z_{M}}_{3}}}        $  (long_name='S114. SS auxiliary export variable 3 (home)           ')
SS_Z_M_3_ST        $  {\overbar{{Z_{M}}_{3}^{\ast}}} $  (long_name='S115. SS auxiliary export variable 3 (foreign)        ')
SS_Z_M_4           $  {\overbar{{Z_{M}}_{4}}}        $  (long_name='S116. SS auxiliary export variable 4 (home)           ')
SS_Z_M_4_ST        $  {\overbar{{Z_{M}}_{4}^{\ast}}} $  (long_name='S117. SS auxiliary export variable 4 (foreign)        ')
SS_Z_M_5           $  {\overbar{{Z_{M}}_{5}}}        $  (long_name='S118. SS auxiliary export variable 5 (home)           ')
SS_Z_M_5_ST        $  {\overbar{{Z_{M}}_{5}^{\ast}}} $  (long_name='S119. SS auxiliary export variable 5 (foreign)        ')
SS_Z_M_6           $  {\overbar{{Z_{M}}_{6}}}        $  (long_name='S120. SS auxiliary export variable 6 (home)           ')
SS_Z_M_6_ST        $  {\overbar{{Z_{M}}_{6}^{\ast}}} $  (long_name='S121. SS auxiliary export variable 6 (foreign)        ')
SS_Z_ST            $  {\overbar{Z^{\ast}}}           $  (long_name='S122. SS aggregate productivity (foreign)             ')
;//-----------------------------------------------------------------------------------------------------------------------')
                       
//Choose calibration option by commenting / uncommenting as desired
                       
//-----------------------------------------//
// Advanced economy calibration            //
//-----------------------------------------//

/*
// ----------------------------------------------------------------------------------------------------------------------|
//  NAME         VALUE         //  #           DESCRIPTION                                |      LaTeX / Sci Word        |
// ---------------------------------------------------------------------------------------|------------------------------|
beta            = 0.9963;      //  1. Deterministic, quarterly discount factor            |  \beta                       |
iota            = 0.23;        //  2. Degree of domestic price indexation (home)          |  \iota                       |
iota_e          = 0.0;         //  3. Weight of exchange rate in wage indexation          |  \iota_{e}                   |
iota_m          = 0.23;        //  4. Degree of export price indexation (home)            |  \iota_{m}                   |
iota_m_ST       = 0.23;        //  5. Degree of export price indexation (foreign)         |  \iota_{m}^{*}               |
iota_w          = 0.5;         //  6. Degree of wage indexation (home)                    |  \iota_{w}                   |
SS_PI_C         = 1.005;       // S1. Steady state inflation (foreign)                    |                              |
s_py            = 0.8;         //  7. Equal s_my. With equal beta implies zero SS NFA     |  \s_{py}                     |
xi_m            = 0.90;        //  8. Calvo probability for exports (home)                |  \xi_{m}                     |
xi_m_ST         = 0.90;        //  9. Calvo probability for exports (foreign)             |  \xi_{m}^{*}                 |
xi_p            = 0.92;        // 10. Calvo prob. for prices of domestic sales (home)     |  \xi_{p}                     |
xi_w            = 0.85;        // 11. Calvo prob. for wages (home)                        |  \xi_{w}                     |
psi             = 0;          // 32. Weight on Kimball (domestic sales, home)           |  \psi                        |
psi_m_ST        = 0;          // 35. Weight on Kimball (exports, foreign)               |  \psi_m^{\ast}               |
omega_f         = 1;         // 29. Domestic ownership share of financiers             |  \omega_{F}                  |
omega_p         = 1;         // 30. Domestic ownership share of exog. financiers       |  \omega_{P}                  |
m_by            = 1000;
// ----------------------------------------------------------------------------------------------------------------------|
*/

//-----------------------------------------//
// Emerging Market calibration             //
//-----------------------------------------//

// ----------------------------------------------------------------------------------------------------------------------|
//  NAME         VALUE         //  #           DESCRIPTION                                |      LaTeX / Sci Word        |
// ---------------------------------------------------------------------------------------|------------------------------|
beta            = 0.9953;      //  1. Deterministic, quarterly discount factor            |  \beta                       |
iota            = 0.75;        //  2. Degree of domestic price indexation (home)          |  \iota                       |
iota_e          = 0*0.25;        //  3. Weight of exchange rate in wage indexation          |  \iota_{e}                   |
iota_m          = 0.75;        //  4. Degree of export price indexation (home)            |  \iota_{m}                   |
iota_m_ST       = 0.75;        //  5. Degree of export price indexation (foreign)         |  \iota_{m}^{*}               |
iota_w          = 0.75;        //  6. Degree of wage indexation (home)                    |  \iota_{w}                   |
SS_PI_C         = 1.01;        // S1. Steady state inflation (foreign)                    |                              |
s_py            = 1.66;        //  7. s_my + 4*0.215 (implies SS NFA = 22% of annual GDP) |  \s_{py}                     |
xi_m            = 0.93;        //  8. Calvo probability for exports (home)                |  \xi_{m}                     |
xi_m_ST         = 0.40;        //  9. Calvo probability for exports (foreign)             |  \xi_{m}^{*}                 |
xi_p            = 0.63;        // 10. Calvo prob. for prices of domestic sales (home)     |  \xi_{p}                     |
xi_w            = 0.81;        // 11. Calvo prob. for wages (home)                        |  \xi_{w}                     |
psi             = -12;          // 32. Weight on Kimball (domestic sales, home)           |  \psi                        |
psi_m_ST        = -12;          // 35. Weight on Kimball (exports, foreign)               |  \psi_m^{\ast}               |
omega_f         = 0.75;         // 29. Domestic ownership share of financiers             |  \omega_{F}                  |
omega_p         = 0.75;         // 30. Domestic ownership share of exog. financiers       |  \omega_{P}                  |
m_by            = 0.1185;
// ----------------------------------------------------------------------------------------------------------------------|


// ----------------------------------------------------------------------------------------------------------------------|
//                 Common Parameters (shared by both the Advanced and Emerging Market calibrations)                      |
// ----------------------------------------------------------------------------------------------------------------------|

// ----------------------------------------------------------------------------------------------------------------------|
//  NAME         VALUE          //  #           DESCRIPTION                               |      LaTeX / Sci Word        |
// ---------------------------------------------------------------------------------------|------------------------------|
alpha           = 0.3;          // 12. Coefficient on capital in the production function  |  \alpha                      |
beta_ST         = 0.9963;       // 13. Foreign discount factor, real int. rate in US=1.5% |  \beta^{*}                   |
cfm_nonfa       = 0;            // 14. Switch moving from price to quantity CFMs          |                              |
chi             = 1;            // 15. Inverse of Frisch Elasticity of Labor Supply       |  \chi                        |
chi_0           = 1;            // 16. Relative weight on labor disutility (home)         |  \chi_{0}                    |
chi_0_ST        = 1;            // 17. Relative weight on labor disutility (foreign)      |  \chi_{0}^{*}                |
elb             = 1;            // 18. Gross nominal int. rate at ELB (home)              |                              |
elb_ST          = 1;            // 19. Gross nominal int. rate at ELB (foreign)           |                              |
eta_0           = 0;            // 20. Private utility from government consumption        |  \eta_{0}                    |
gamma_0         = 0.06;         // 21. To get FXI effect in line with Adler-Lisack-Mano   |  \gamma_{0}                  |
gamma_1         = 0;            // 22. Gabaix-Maggiori Gamma semi-elast. wrt. ER variance |  \gamma_{1}                  |
iota_ST         = 0.23;         // 23. Domestic price indexation (foreign, SW AER 2007)   |  \iota^{*}                   |
iota_w_ST       = 0.5;          // 24. Degree of wage indexation (foreign)                |  \iota_{w}^{*}               |
nu              = 1;            // 25. No long memory in wage indexation term (home)      |  \nu                         |
nu_ST           = 1;            // 26. No long memory in wage indexation term (foreign)   |  \nu^{*}                     |
omega_c         = 0.29;         // 27. Consumption share of domestic goods (home C)       |  \omega_{C}                  |
omega_g         = 0.1;          // 28. Govt. cons. share of domestic goods (home G)       |  \omega_{G}                  |
omega_b         = 1;            // 31. Domestic ownership share of banks                  |  \omega_{B}                  |
psi_ST          = 0;            // 33. Weight on Kimball (domestic sales, foreign)        |  \psi^{\ast}                 |
psi_m           = 0;            // 34. Weight on Kimball (exports, home)                  |  \psi_m                      |
psi_pi          = 0;            // 36. Coeff. on PI_C in domestic Taylor rule             |  \psi_{\pi}                  |
psi_pi_ST       = 0;            // 37. Coeff. on PI_C_ST in foreign Taylor rule           |  \psi_{\pi}^{\ast}           |
psi_pid         = 1.5;          // 38. Coeff. on PI_D in domestic Taylor rule             |  \psi_{\pi_d}                |
psi_pid_ST      = 1.5;          // 39. Coeff. on PI_D_ST in foreign Taylor rule           |  \psi_{\pi_d}^{\ast}         |
psi_x           = 0.125/2;      // 40. Coeff. on output gap in domestic Taylor rule       |  \psi_{x}                    |
psi_x_ST        = 0.125/2;      // 41. Coeff. on output gap in foreign Taylor rule        |  \psi_{x}^{\ast}             |
psi_i           = 0;            // 42. Policy rate smoothing in domestic Taylor rule      |  \psi_{i}                    |
psi_i_ST        = 0;            // 43. Policy rate smoothing in foreign Taylor rule       |  \psi_{i}^{\ast}             |
psi_theta       = 0;            // 44. Coeff. on spread in domestic Taylor rule           |  \psi_{\theta}               |
rho_c           = -5;           // 45. Elasticity of substitution (home - foreign C)      |  \rho_{C}                    |
rho_c_ST        = -5;           // 46. Elasticity of substitution (foreign - home C)      |  \rho_{C}^{\ast}             |
rho_g           = -5;           // 47. Elasticity of substitution (home - foreign G)      |  \rho_{G}                    |
rho_g_ST        = -5;           // 48. Elasticity of substitution (foreign - home G)      |  \rho_{G}^{\ast}             |
rho_nu          = 0.95;         // 49. Persistence of demand shock (home)                 |  \rho_{\nu}                  |
rho_nu_ST       = 0.95;         // 50. Persistence of demand shock (foreign)              |  \rho_{\nu}^{\ast}           |
rho_tau_c       = 0.95;         // 51. Persistence of consumption taxation (home)         |  \rho_{\tau_{C}}             |
rho_tau_c_ST    = 0.95;         // 52. Persistence of consumption taxation (foreign)      |  \rho_{\tau_{C}}^{\ast}      |
rho_tau_f       = 0.9;          // 53. Persistence of capital inflow taxation (home)      |  \rho_{\tau_{f}}             |
rho_tau_n       = 0.95;         // 54. Persistence of labor taxation (home)               |  \rho_{\tau_{N}}             |
rho_tau_n_ST    = 0.95;         // 55. Persistence of labor taxation (foreign)            |  \rho_{\tau_{N}}^{\ast}      |
rho_varsigma    = 0.95;         // 56. Persistence of preference shock (home)             |  \rho_{\varsigma}            |
rho_varsigma_ST = 0.95;         // 57. Persistence of preference shock (foreign)          |  \rho_{\varsigma}^{\ast}     |
rho_upsilon     = 0.0;          // 58. Persistence of domestic markup shock (home)        |  \rho_{\upsilon}             |
rho_upsilon_ST  = 0.0;          // 59. Persistence of domestic markup shock (foreign)     |  \rho_{\upsilon}^{\ast}      |
rho_upsilon_m   = 0.0;          // 60. Persistence of import markup shock (home)          |  \rho_{\upsilon_m}           |
rho_upsilon_m_ST= 0.0;          // 61. Persistence of import markup shock (foreign)       |  \rho_{\upsilon_m}^{\ast}    |
rho_upsilon_w   = 0.0;          // 62. Persistence of wage markup shock (home)            |  \rho_{\upsilon_w}           |
rho_upsilon_w_ST= 0.0;          // 63. Persistence of wage markup shock (foreign)         |  \rho_{\upsilon_w}^{\ast}    |
rho_e_i         = 0.0;          // 64. Persistence of monetary policy shock (home)        |  \rho_{e,i}                  |
rho_e_i_ST      = 0.0;          // 65. Persistence of monetary policy shock (foreign)     |  \rho_{e,i}^{\ast}           |
rho_z           = 0.95;         // 66. Persistence of aggregate productivity shock (home) |  \rho_{Z}                    |
rho_z_ST        = 0.95;         // 67. Persistence of aggregate prod. shock (foreign)     |  \rho_{Z}^{\ast}             |
s_gy            = 0.14;         // 68. Share of government consumption in GDP (home)      |  s_{gy}                      |
s_gy_ST         = 0.15;         // 69. Share of government consumption in GDP (foreign)   |  s_{gy}^{\ast}               |
s_my            = 0.8;          // 70. SS FX reserves to annual GDP = 20%                 |  \s_{my}                     |
sigma           = 1;            // 71. Intertemporal elasticity of substitution           |  \sigma                      |
spill_i         = 0;            // 72. Parameter #1 controlling spillovers                |                              |
spill_upsilon_w = 0;            // 73. Parameter #2 controlling spillovers                |                              |
spill_varsigma  = 0;            // 74. Parameter #3 controlling spillovers                |                              |
spill_z         = 0;            // 75. Parameter #4 controlling spillovers                |                              |
theta_p         = 0.2;          // 76. Steady state net markup in product markets         |  \theta_{p}                  |
theta_w         = 0.5;          // 77. Steady state net markup in labor market            |  \theta_{w}                  |
tau_p           = 0*theta_p;      // 78. Subsidy to production                              |  \tau_{p}                    |
tau_w           = 0*theta_w;      // 79. Subsidy to labor                                   |  \tau_{w}                    |
var_e           = 1;            // 80. Conditional variance of ER                         |                              |
varkappa        = 0;            // 81. Habit parameter (home)                             |  \varkappa                   |
varkappa_ST     = 0;            // 82. Habit parameter (foreign)                          |  \varkappa^{\ast}            |
varrho_m        = 0.90;         // 83. Persistence of FXI                                 |  \varrho_{m}                 |
varrho_p        = 0.95;         // 84. Persistence of portfolio shock                     |  \varrho_{p}                 |
varrho_g        = 0.796;        // 85. Persistence of government spending (home)          |  \varrho_{G}                 |
varrho_g_ST     = 0.967;        // 86. Persistence of government spending (foreign)       |  \varrho_{G}^{\ast}          |
xi_p_ST         = 0.92;         // 87. Calvo prob. for prices (foreign, as in SW AER 2007)|  \xi_{p}^{\ast}              |
xi_w_ST         = 0.85;         // 88. Calvo prob. for wages (foreign, as in SW AER 2007) |  \xi_{w}^{\ast}              |
zeta            = 1/100;        // 89. Relative size of home economy                      |  \zeta                       |
zeta_ST         = 1-zeta;       // 90. Relative size of foreign economy                   |  \zeta^{\ast}                |
// ----------------------------------------------------------------------------------------------------------------------|
                                   
//-----------------------------------------//
// Aggressive monetary policy option       //
//-----------------------------------------//
                                   
/*
// ----------------------------------------------------------------------------------------------------------------------|
psi_pi          = 3;           //  36. Coeff. on PI_C in domestic Taylor rule             |  \psi_{\pi}                  |
psi_pid         = 2;           //  38. Coeff. on PI_D in domestic Taylor rule             |  \psi_{\pi_d}                |
// ----------------------------------------------------------------------------------------------------------------------|
*/

//-----------------------------------------//
// Vulnerable EM calibration               //
//-----------------------------------------//
/*
// ----------------------------------------------------------------------------------------------------------------------|
gamma_0         = 0.08;        //  21. 33% greater market shallowness parameter           |  \gamma_{0}                  |
s_py        = s_my + 4*0.45;   //   7. 45% NFL due to exogenous portfolio decisions       |  \s_{py}                     |
// ----------------------------------------------------------------------------------------------------------------------|
*/

//Shock spillover parameters
spill_i = 0; 
spill_z = 0; 
spill_varsigma = 0; 
spill_upsilon_w = 0;

//Parameters from external calibration
varrho_g        = 0.796; 
varrho_g_ST     = 0.973; 

//Parameters from moment matching
rho_varsigma_ST = 0.92656;
rho_z_ST        = 0.41138;
rho_varsigma   = 0.72868;
rho_z          = 0.36637;
varrho_p       = 0.95;
spill_z        = 0.59025;
spill_varsigma = 0.15238;

//Parameters in the FXI and CFM rules
parameters ppsim_bp ppsim_theta ppsif_b;
ppsim_bp = 0;
ppsim_theta = 0;
ppsif_b = 0;

//Overwrite parameters for stochastic simulations
//@#include "parameterization_stoch_sims.m"

// ----------------------------------------------------------------------------------------------------------------------|
//                                                  STEADY STATE DERIVATION                                              |
// ----------------------------------------------------------------------------------------------------------------------|
SS_E_I              = 0;       // S2.
SS_E_I_ST           = 0;       // S3.
SS_GAM_CD           = 1;       // S4.
SS_GAM_CD_ST        = 1;       // S5.
SS_GAM_CM           = 1;       // S6.
SS_GAM_CM_ST        = 1;       // S7.
SS_GAM_GD           = 1;       // S8.
SS_GAM_GD_ST        = 1;       // S9.
SS_GAM_GM           = 1;       //S10.
SS_GAM_GM_ST        = 1;       //S11.
SS_GAM_MD           = 1;       //S12.
SS_GAM_MD_ST        = 1;       //S13.
SS_NU               = 0*0.01;    //S14.
SS_NU_ST            = 0*0.01;    //S15.
SS_P_AMP_D          = 1;       //S16.
SS_P_AMP_D_ST       = 1;       //S17.
SS_P_AMP_M          = 1;       //S18.
SS_P_AMP_M_ST       = 1;       //S19.
SS_P_TIL_D          = 1;       //S20.
SS_P_TIL_D_ST       = 1;       //S21.
SS_P_TIL_M          = 1;       //S22.
SS_P_TIL_M_ST       = 1;       //S23.
SS_PI_C_ST          = 1.005;   //S24.
SS_Q                = 1;       //S25.
SS_TAU_C            = 0*0.15;    //S26.
SS_TAU_C_ST         = 0*0.15;    //S27.
SS_TAU_F            = 0;       //S28.
SS_TAU_N            = 0*0.15;    //S29.
SS_TAU_N_ST         = 0*0.15;    //S30.
SS_UPSILON          = 1+tau_p; //S31.
SS_UPSILON_ST       = 1+tau_p; //S32.
SS_UPSILON_M_ST     = 1+tau_p; //S33.
SS_UPSILON_M        = 1+tau_p; //S34.
SS_UPSILON_W        = 1+tau_w; //S35.
SS_UPSILON_W_ST     = 1+tau_w; //S36.
SS_VARSIGMA         = 1;       //S37.
SS_VARSIGMA_ST      = 1;       //S38.
SS_VARTHETA         = 1;       //S39.
SS_VARTHETA_ST      = 1;       //S40.
SS_VARTHETA_M       = 1;       //S41.
SS_VARTHETA_M_ST    = 1;       //S42.
SS_W_AMP_U          = 1;       //S43.
SS_W_AMP_U_ST       = 1;       //S44.
SS_Z                = 1;       //S45.
SS_Z_ST             = 1;       //S46.
SS_Z_4              = 1;       //S47.
SS_Z_4_ST           = 1;       //S48.
SS_Z_5              = 1;       //S49.
SS_Z_5_ST           = 1;       //S50.
SS_Z_6              = 1;       //S51.
SS_Z_6_ST           = 1;       //S52.
SS_Z_M_4            = 1;       //S53.
SS_Z_M_4_ST         = 1;       //S54.
SS_Z_M_5            = 1;       //S55.
SS_Z_M_5_ST         = 1;       //S56.
SS_Z_M_6            = 1;       //S57.
SS_Z_M_6_ST         = 1;       //S58.
                                  
                                  
//This is the "contingent" part ofthe Steady State that may change when any of the underlying parameter values change
                                  
//----------------------------------------//
SS_GAMMA            = gamma_0*var_e^gamma_1;                                                   //S59.
k_n                 = ((1+theta_p)/(1+tau_p)*(1-beta)/(alpha*beta))^(1/(alpha-1));             // 91.
SS_MC_D             = (1+tau_p)/(1+theta_p);                                                   //S60.
SS_MC_D_ST          = (1+tau_p)/(1+theta_p);                                                   //S61.
                                  
//----------------------------------------//
k_n_ST              = ((1+theta_p)/(1+tau_p)*(1-beta_ST)/(alpha*beta_ST))^(1/(alpha-1));       // 92.  
b_y                 = (s_my - s_py) - 1/SS_GAMMA*(1-SS_TAU_F-beta/beta_ST); // 93.    
c_y                 = 1 - s_gy - b_y*(1-(1-omega_f)*(1-SS_TAU_F)*1/beta-omega_f*1/beta_ST) - ((1-SS_TAU_F)*1/beta-1/beta_ST)*((omega_f-omega_p)*s_py+(1-omega_f)*s_my);                                           //  94.
SS_N                = (1/((c_y + eta_0*s_gy)*(1-varkappa-SS_NU))*((1+tau_p)*(1+tau_w)*(1-alpha)/(1+theta_p)*(1/chi_0/(1+theta_w))*(1-SS_TAU_N)/(1+SS_TAU_C))^(sigma)*(k_n)^(alpha*(sigma-1)))^(1/(1+chi*sigma));  // S62.
k                   = k_n*SS_N;                                                                // 95.
SS_Y                = k_n^(alpha)*SS_N;                                                        //S63.
SS_C                = c_y*SS_Y;                                                                //S64.
                                  
//----------------------------------------//
//            Solve for SS N              //
//----------------------------------------//
SS_N_ST             = QMIPF_solve_SS(zeta,zeta_ST,s_gy,s_gy_ST,SS_Y,SS_C,chi,sigma,eta_0,k_n_ST,alpha,theta_p,chi_0_ST,SS_TAU_N_ST,SS_TAU_C_ST,varkappa_ST,SS_NU_ST,SS_N,theta_w,tau_p,tau_w); // S65.
                                  
k_ST                = k_n_ST*SS_N_ST;                                                                                                                                        // 96.
SS_Y_ST             = k_n_ST^(alpha)*SS_N_ST;                                                                                                                                // S66.
SS_C_ST             = zeta/zeta_ST*((1-s_gy)*SS_Y-SS_C) + (1-s_gy_ST)*k_n_ST^alpha*SS_N_ST;                                                                                  // S67.
SS_I                = SS_PI_C/beta;                                                                                                                                          // S68.
SS_I_ST             = SS_PI_C_ST/beta_ST;                                                                                                                                    // S69.
SS_IB               = SS_I;                                                                                                                                                  // S70.
SS_THETA            = 0;                                                                                                                                                     // S71.
SS_PI_D             = SS_PI_C;                                                                                                                                               // S72.
SS_PI_D_ST          = SS_PI_C_ST;                                                                                                                                            // S73.
SS_PI_M             = SS_PI_C;                                                                                                                                               // S74.
SS_PI_M_ST          = SS_PI_C_ST;                                                                                                                                            // S75.
SS_W_C              = (1+tau_p)/(1+theta_p)*(1-alpha)*k_n^(alpha);                                                                                                           // S76.
SS_W_C_ST           = (1+tau_p)/(1+theta_p)*(1-alpha)*k_n_ST^(alpha);                                                                                                        // S77.
SS_B                = b_y*SS_Y;                                                                                                                                              // S78.
SS_B_P              = s_py*SS_Y;                                                                                                                                             // S79.
SS_B_M              = s_my*SS_Y;                                                                                                                                             // S80.
SS_B_F              = -SS_B - SS_B_P + SS_B_M;                                                                                                                               // S81.
SS_G                = s_gy*SS_Y;                                                                                                                                             // S82.
SS_G_ST             = s_gy_ST*SS_Y_ST;                                                                                                                                       // S83.
SS_C_TIL            = SS_C+eta_0*SS_G;                                                                                                                                       // S84.
SS_C_TIL_ST         = SS_C_ST+eta_0*SS_G_ST;                                                                                                                                 // S85.
SS_LAM              = (SS_C_TIL-varkappa*SS_C_TIL-SS_C_TIL*SS_NU)^(-1/sigma)/(1+SS_TAU_C);                                                                                   // S86.
SS_LAM_ST           = (SS_C_TIL_ST-varkappa_ST*SS_C_TIL_ST-SS_C_TIL_ST*SS_NU_ST)^(-1/sigma)/(1+SS_TAU_C_ST);                                                                 // S87.
SS_W_TIL_C          = SS_W_C;                                                                                                                                                // S88.
SS_W_TIL_C_ST       = SS_W_C_ST;                                                                                                                                             // S89.
SS_PI_W             = SS_PI_C;                                                                                                                                               // S90.
SS_PI_W_ST          = SS_PI_C_ST;                                                                                                                                            // S91.
                                                                                                                                                                             
//-----------------------------------------//                                                                                                                                
// Normalization factors for auxiliary variables from sticky wage block (needed for lmmcp)                                                                                   
z7_corr             = chi_0*SS_W_C^((1+theta_w)/theta_w*(1+chi))*SS_N^(1+chi);                                                                                               // 97.
z8_corr             = SS_LAM*(1-SS_TAU_N)*SS_W_C^((1+theta_w)/theta_w)*SS_N;                                                                                                 // 98.
z7_corr_ST          = chi_0_ST*SS_W_C_ST^((1+theta_w)/theta_w*(1+chi))*SS_N_ST^(1+chi);                                                                                      // 99.
z8_corr_ST          = SS_LAM_ST*(1-SS_TAU_N_ST)*SS_W_C_ST^((1+theta_w)/theta_w)*SS_N_ST;                                                                                     //100.
//-----------------------------------------//
                                  
SS_Z_7              = SS_VARSIGMA*chi_0*SS_W_C^((1+theta_w)/theta_w*(1+chi))*SS_N^(1+chi)/(1-beta*xi_w) /z7_corr;                                                            // S92.  
SS_Z_7_ST           = SS_VARSIGMA_ST*chi_0_ST*SS_W_C_ST^((1+theta_w)/theta_w*(1+chi))*SS_N_ST^(1+chi)/(1-beta*xi_w_ST) /z7_corr_ST;                                          // S93.  
SS_Z_8              = SS_VARSIGMA*SS_LAM*(1-SS_TAU_N)*(1+tau_w)*SS_W_C^((1+theta_w)/theta_w)*SS_N/(1-beta*xi_w) /z8_corr;                                                              // S94.  
SS_Z_8_ST           = SS_VARSIGMA_ST*SS_LAM_ST*(1-SS_TAU_N_ST)*(1+tau_w)*SS_W_C_ST^((1+theta_w)/theta_w)*SS_N_ST/(1-beta*xi_w_ST) /z8_corr_ST;                                         // S95.  
SS_PI_P             = SS_PI_D;                                                                                                                                               // S96.  
SS_PI_P_ST          = SS_PI_D_ST;                                                                                                                                            // S97.  
SS_PI_PM            = SS_PI_D;                                                                                                                                               // S98.  
SS_PI_PM_ST         = SS_PI_D_ST;                                                                                                                                            // S99.  
SS_U                = ( 1/(1-1/sigma)*(SS_C_TIL-varkappa*SS_C_TIL-SS_C_TIL*SS_NU)^(1-1/sigma) - chi_0*SS_W_AMP_U*SS_N^(1+chi)/(1+chi) )/(1-beta);                            //S100.
SS_U_ST             = ( 1/(1-1/sigma)*(SS_C_TIL_ST-varkappa_ST*SS_C_TIL_ST-SS_C_TIL_ST*SS_NU_ST)^(1-1/sigma) - chi_0_ST*SS_W_AMP_U_ST*SS_N_ST^(1+chi)/(1+chi) )/(1-beta_ST); //S101.
// Overwrite for log utility
if sigma==1
SS_U                = ( log(SS_C_TIL-varkappa*SS_C_TIL-SS_C_TIL*SS_NU) - chi_0*SS_W_AMP_U*SS_N^(1+chi)/(1+chi) )/(1-beta);                                                   //S100.
SS_U_ST             = ( log(SS_C_TIL_ST-varkappa_ST*SS_C_TIL_ST-SS_C_TIL_ST*SS_NU_ST) - chi_0_ST*SS_W_AMP_U_ST*SS_N_ST^(1+chi)/(1+chi) )/(1-beta_ST);                        //S101.
end

//-----------------------------------------//
// Here we set the home bias parameters for the foreign economy so that they are consistent with Q=1
s_omega_g_c_ST      = 0.1/0.29; //When adjusting we will want to preserve a reasonable ratio                                                                                 //101.
omega_c_ST          = zeta/zeta_ST* (SS_Y-(1-omega_c)*SS_C-(1-omega_g)*SS_G) / (SS_C_ST+s_omega_g_c_ST*SS_G_ST);                                                             //102.
omega_g_ST          = s_omega_g_c_ST*omega_c_ST;                                                                                                                             //103.
                                                                                                                                                                             
//-----------------------------------------//                                                                                                                                
SS_Y_M              = zeta/zeta_ST*(omega_c*SS_C+omega_g*SS_G);                                                                                                              //S102.
SS_Y_M_ST           = zeta_ST/zeta*(omega_c_ST*SS_C_ST+omega_g_ST*SS_G_ST);                                                                                                  //S103.
SS_Y_D              = SS_Y-SS_Y_M_ST;                                                                                                                                        //S104.
SS_Y_D_ST           = SS_Y_ST-SS_Y_M;                                                                                                                                        //S105.
SS_Z_1              = (1+psi)*(1+theta_p)/((1-beta*xi_p)*(1+psi+psi*theta_p))*SS_LAM*SS_Y_D*SS_MC_D;                                                               //S106.
SS_Z_1_ST           = (1+psi_ST)*(1+theta_p)/((1-beta_ST*xi_p_ST)*(1+psi_ST+psi_ST*theta_p))*SS_LAM_ST*SS_Y_D_ST*SS_MC_D_ST;                                       //S107.
SS_Z_2              = (1+tau_p)*SS_LAM*SS_Y_D/(1-beta*xi_p);                                                                                                                           //S108.
SS_Z_2_ST           = (1+tau_p)*SS_LAM_ST*SS_Y_D_ST/(1-beta_ST*xi_p_ST);                                                                                                               //S109.
SS_Z_3              = (1+tau_p)*(psi*theta_p)/(1+psi+psi*theta_p)*SS_LAM*SS_Y_D/(1-beta*xi_p);                                                                                         //S110.
SS_Z_3_ST           = (1+tau_p)*(psi_ST*theta_p)/(1+psi_ST+psi_ST*theta_p)*SS_LAM_ST*SS_Y_D_ST/(1-beta_ST*xi_p_ST);                                                                    //S111.
SS_Z_M_1            = (1+psi_m)*(1+theta_p)/((1-beta*xi_m)*(1+psi_m+psi_m*theta_p))*SS_LAM*SS_Y_M_ST*SS_MC_D;                                                      //S112.
SS_Z_M_1_ST         = (1+psi_m_ST)*(1+theta_p)/((1-beta_ST*xi_m_ST)*(1+psi_m_ST+psi_m_ST*theta_p))*SS_LAM_ST*SS_Y_M*SS_MC_D_ST;                                    //S113.
SS_Z_M_2            = SS_LAM*SS_Y_M_ST*(1+tau_p)/(1-beta*xi_m);                                                                                                                        //S114.
SS_Z_M_2_ST         = SS_LAM_ST*SS_Y_M*(1+tau_p)/(1-beta_ST*xi_m_ST);                                                                                                                  //S115.
SS_Z_M_3            = (psi_m*theta_p)/(1+psi_m+psi_m*theta_p)*SS_LAM*SS_Y_M_ST*(1+tau_p)/(1-beta*xi_m);                                                                                //S116.
SS_Z_M_3_ST         = (psi_m_ST*theta_p)/(1+psi_m_ST+psi_m_ST*theta_p)*SS_LAM_ST*SS_Y_M*(1+tau_p)/(1-beta_ST*xi_m_ST);                                                                 //S117.
SS_M_C              = omega_c*SS_C;                                                                                                                                          //S118.                                       
SS_M_C_ST           = omega_c_ST*SS_C_ST;                                                                                                                                    //S119.                                       
SS_M_G              = omega_g*SS_G;                                                                                                                                          //S120.                                       
SS_M_G_ST           = omega_g_ST*SS_G_ST;                                                                                                                                    //S121.                                       
                                                                                                                                            
m                   = -SS_B/SS_Y + m_by*4;  // Debt limit 12 percent above steady state                                                                                      //104.                                        
SS_BLIM             = SS_B + m*SS_Y;                                                                                                                                         //S122.                                       
                                                                                                                                            
                                                                                                                                            
//-------------------------------------------------------------------------------------------------------------------------)                
// 3. Model declaration                                                                                                                     
//-------------------------------------------------------------------------------------------------------------------------)                
model;                                                                                                                                   
//-------------------------------------------------------------------------------------------------------------------------)                
// Home Block                                                                                                                               
//-------------------------------------------------------------------------------------------------------------------------)                
                                                                                                                                            
[name='Marginal utility definition (optimality condition with respect to consumption)']                                                     
LAM=(C_TIL-varkappa*C_TIL(-1)-SS_C_TIL*NU)^(-1/sigma)/(1+TAU_C); // Equation 1

[name='Euler equation (optimality condition with respect to domestic bond holdings)']
LAM=beta*(VARSIGMA(+1)/VARSIGMA)*(IB/PI_C(+1))*LAM(+1); // Equation 2
                                                                    
[name='Optimal reset wage']
W_TIL_C^(1+(1+theta_w)/theta_w*chi)=(1+theta_w)*Z_7/Z_8*z7_corr/z8_corr; // Equation 3

[name='Recursive law of motion for Z_7']
Z_7=VARSIGMA*chi_0*W_C^((1+theta_w)/theta_w*(1+chi))*N^(1+chi)/z7_corr +beta*xi_w*(PI_W(+1)/PI_C(+1))^(-(1+theta_w)/theta_w*(1+chi))*Z_7(+1); // Equation 4

[name='Recursive law of motion for Z_8']
Z_8=VARSIGMA*LAM*(1-TAU_N)*UPSILON_W*W_C^((1+theta_w)/theta_w)*N/z8_corr +beta*xi_w*(PI_W(+1)/PI_C(+1))^(-1/theta_w)*Z_8(+1); // Equation 5

[name='Aggregate real wage']
W_C^(-1/theta_w)=(1-xi_w)*W_TIL_C^(-1/theta_w)+xi_w*(PI_W/PI_C*W_C(-1))^(-1/theta_w); // Equation 6

[name='Wage indexation scheme']
PI_W=PI_W(-1)^(1-nu)*SS_PI_C^(nu*(1-iota_w))*(PI_C(-1)^(1-iota_e)*(Q(-1)/Q(-2)*PI_C(-1)/PI_C_ST(-1)*SS_PI_C_ST)^iota_e)^(nu*iota_w); // Equation 7

[name='Wage dispersion for labor disutility aggregation']
W_AMP_U=(1-xi_w)*(W_TIL_C/W_C)^(-(1+theta_w)/theta_w*(1+chi))+xi_w*W_AMP_U(-1)*(PI_W/PI_C*W_C(-1)/W_C)^(-(1+theta_w)/theta_w*(1+chi)); // Equation 8
                
[name='Link between effective consumption and actual private and government consumption']
C_TIL=C+eta_0*G; // Equation 9
                
[name='Expression for real marginal cost in terms of domestic goods prices']
MC_D=1/(1-alpha)*W_C*GAM_CD*(N/k)^(alpha)*(1/Z^(1-alpha)); // Equation 10

[name='Resetting firms optimality condition']
Z_1=Z_2*P_TIL_D-Z_3*P_TIL_D^(1+(1+theta_p)/theta_p*(1+psi)); // Equation 11

[name='Recursive law of motion for Z_1']
Z_1=(1+psi)*(1+theta_p)/(1+psi+psi*theta_p)*VARSIGMA*LAM*Y_D*VARTHETA^((1+theta_p)/theta_p*(1+psi))*MC_D/GAM_CD+beta*xi_p*(PI_P(+1)/PI_D(+1))^(-(1+theta_p)/theta_p*(1+psi))*Z_1(+1); // Equation 12

[name='Recursive law of motion for Z_2']
Z_2=VARSIGMA*LAM*UPSILON*Y_D/GAM_CD*VARTHETA^((1+theta_p)/theta_p*(1+psi))+beta*xi_p*(PI_P(+1)/PI_D(+1))^(-(1+psi+psi*theta_p)/theta_p)*Z_2(+1); // Equation 13

[name='Recursive law of motion for Z_3']
Z_3=(psi*theta_p)/(1+psi+psi*theta_p)*VARSIGMA*LAM*UPSILON*Y_D/GAM_CD+beta*xi_p*PI_P(+1)/PI_D(+1)*Z_3(+1); // Equation 14

[name='Price indexation scheme, domestic sales']
PI_P=SS_PI_D^(1-iota)*PI_D(-1)^iota; // Equation 15

[name='Aggregate production function']
P_AMP_D*Y_D+P_AMP_M_ST*Y_M_ST=k^(alpha)*(Z*N)^(1-alpha); // Equation 16

[name='Law of motion for domestic price dispersion term']
P_AMP_D=VARTHETA^((1+theta_p)/theta_p*(1+psi))/(1+psi)*Z_4^(-(1+theta_p)/theta_p*(1+psi))+psi/(1+psi); // Equation 17

[name='Recursive law of motion for auxiliary terms governing domestic price dispersion']
Z_4^(-(1+theta_p)*(1+psi)/theta_p)=(1-xi_p)*P_TIL_D^(-(1+theta_p)*(1+psi)/theta_p)+xi_p*(PI_P/PI_D*Z_4(-1))^(-(1+theta_p)*(1+psi)/theta_p); // Equation 18

[name='Law of motion for export price dispersion term']
P_AMP_M_ST=VARTHETA_M^((1+theta_p)/theta_p*(1+psi_m))/(1+psi_m)*Z_M_4^(-(1+theta_p)/theta_p*(1+psi_m))+psi_m/(1+psi_m); // Equation 19

[name='Recursive law of motion for auxiliary terms governing export price dispersion']
Z_M_4^(-(1+theta_p)*(1+psi_m)/theta_p)=(1-xi_m)*P_TIL_M_ST^(-(1+theta_p)*(1+psi_m)/theta_p)+xi_m*(PI_PM_ST/PI_M_ST*Z_M_4(-1))^(-(1+theta_p)*(1+psi_m)/theta_p); // Equation 20

[name='Zero profit condition for final goods producers']
VARTHETA=1+psi-psi*Z_5; // Equation 21

[name='Recursive law of motion for auxiliary term']
Z_5=(1-xi_p)*P_TIL_D+xi_p*PI_P/PI_D*Z_5(-1); // Equation 22

[name='Definition of the aggregate price index']
VARTHETA=Z_6; // Equation 23

[name='Recursive law of motion for auxiliary term']
Z_6^(-(1+psi+psi*theta_p)/theta_p)=(1-xi_p)*P_TIL_D^(-(1+psi+psi*theta_p)/theta_p)+xi_p*(PI_P/PI_D*Z_6(-1))^(-(1+psi+psi*theta_p)/theta_p); // Equation 24

[name='Aggregate resource constraint for domestic sales']
Y_D=(1-omega_c)*GAM_CD^((1+rho_c)/rho_c)*C+(1-omega_g)*GAM_GD^((1+rho_g)/rho_g)*G; // Equation 25

[name='Resetting firms optimality condition for export sales']
Z_M_1=Z_M_2*P_TIL_M_ST-Z_M_3*P_TIL_M_ST^(1+(1+theta_p)/theta_p*(1+psi_m)); // Equation 26

[name='Recursive law of motion for Z_M_1']
Z_M_1=(1+psi_m)*(1+theta_p)/(1+psi_m+psi_m*theta_p)*VARSIGMA*LAM*Y_M_ST*VARTHETA_M^((1+theta_p)/theta_p*(1+psi_m))*MC_D/GAM_CD+beta*xi_m*(PI_PM_ST(+1)/PI_M_ST(+1))^(-(1+theta_p)/theta_p*(1+psi_m))*Z_M_1(+1); // Equation 27

[name='Recursive law of motion for Z_M_2']
Z_M_2=VARSIGMA*LAM*UPSILON_M_ST*Y_M_ST/GAM_CM_ST*Q*VARTHETA_M^((1+theta_p)/theta_p*(1+psi_m))+beta*xi_m*(PI_PM_ST(+1)/PI_M_ST(+1))^(-(1+psi_m+psi_m*theta_p)/theta_p)*Z_M_2(+1); // Equation 28

[name='Recursive law of motion for Z_M_3']
Z_M_3=(psi_m*theta_p)/(1+psi_m+psi_m*theta_p)*VARSIGMA*LAM*UPSILON_M_ST*Y_M_ST/GAM_CM_ST*Q+beta*xi_m*PI_PM_ST(+1)/PI_M_ST(+1)*Z_M_3(+1); // Equation 29

[name='Price indexation scheme, exports']
PI_PM_ST=SS_PI_M_ST^(1-iota_m)*PI_M_ST(-1)^iota_m; // Equation 30

[name='Zero profit condition for final export goods producers']
VARTHETA_M=1+psi_m-psi_m*Z_M_5; // Equation 31

[name='Recursive law of motion for export auxiliary term']
Z_M_5=(1-xi_m)*P_TIL_M_ST+xi_m*PI_PM_ST/PI_M_ST*Z_M_5(-1); // Equation 32

[name='Definition of the aggregate price index for exports']
VARTHETA_M=Z_M_6; // Equation 33

[name='Recursive law of motion for auxiliary term']
Z_M_6^(-(1+psi_m+psi_m*theta_p)/theta_p)=(1-xi_m)*P_TIL_M_ST^(-(1+psi_m+psi_m*theta_p)/theta_p)+xi_m*(PI_PM_ST/PI_M_ST*Z_M_6(-1))^(-(1+psi_m+psi_m*theta_p)/theta_p); // Equation 34

[name='Aggregate resource constraint for exports']
Y_M_ST=zeta_ST/zeta*(omega_c_ST*(GAM_CM_ST)^((1+rho_c_ST)/rho_c_ST)*C_ST+omega_g_ST*(GAM_GM_ST)^((1+rho_g_ST)/rho_g_ST)*G_ST); // Equation 35

[name='Aggregate to domestically produced price ratio']
GAM_CD=((1-omega_c)+omega_c*(GAM_MD)^(-1/rho_c))^(-rho_c); // Equation 36

[name='Government to domestically produced price ratio']
GAM_GD=((1-omega_g)+omega_g*(GAM_MD)^(-1/rho_g))^(-rho_g); // Equation 37

[name='Foreign aggregate to imported price ratio']
GAM_CM_ST=((1-omega_c_ST)*GAM_MD_ST^(1/rho_c_ST)+omega_c_ST)^(-rho_c_ST); // Equation 38

[name='Foreign government to imported price ratio']
GAM_GM_ST=((1-omega_g_ST)*GAM_MD_ST^(1/rho_g_ST)+omega_g_ST)^(-rho_g_ST); // Equation 39

[name='Domestic consumer price inflation']
PI_C=GAM_CD/GAM_CD(-1)*PI_D; // Equation 40

[name='Definition of domestic import price inflation']
GAM_MD/GAM_MD(-1)=PI_M/PI_D; // Equation 41

[name='Imported component of consumption']
M_C = omega_c*(GAM_CM)^((1+rho_c)/rho_c)*C; // Equation 42

[name='Imported component of government spending']
M_G = omega_g*(GAM_GM)^((1+rho_g)/rho_g)*G; // Equation 43

[name='Determination of home monetary policy']
I=max(elb,psi_theta*THETA +(1-psi_i)*(SS_I+psi_pi*(PI_C-SS_PI_C)+psi_pid*(PI_D-SS_PI_D)+psi_x*(Y/Y_POT-1))+psi_i*I(-1)+E_I); // Equation 44

//-------------------------------------------------------------------------------------------------------------------------)
// Foreign Block
//-------------------------------------------------------------------------------------------------------------------------)

[name='Foreign marginal utility definition (optimality condition with respect to consumption)']
LAM_ST=(C_TIL_ST-varkappa_ST*C_TIL_ST(-1)-SS_C_TIL_ST*NU_ST)^(-1/sigma)/(1+TAU_C_ST); // Equation 45

[name='Foreign Euler equation (optimality condition with respect to domestic bond holdings)']
LAM_ST=beta_ST*(VARSIGMA_ST(+1)/VARSIGMA_ST)*I_ST/PI_C_ST(+1)*LAM_ST(+1); // Equation 46

[name='Foreign optimal reset wage']
W_TIL_C_ST^(1+(1+theta_w)/theta_w*chi)=(1+theta_w)*Z_7_ST/Z_8_ST *z7_corr_ST/z8_corr_ST; // Equation 47

[name='Foreign recursive law of motion forZ_7_ST']
Z_7_ST=VARSIGMA_ST*chi_0_ST*W_C_ST^((1+theta_w)/theta_w*(1+chi))*N_ST^(1+chi)/z7_corr_ST +beta*xi_w_ST*(PI_W_ST(+1)/PI_C_ST(+1))^(-(1+theta_w)/theta_w*(1+chi))*Z_7_ST(+1); // Equation 48

[name='Foreign recursive law of motion for Z_8_ST']
Z_8_ST=VARSIGMA_ST*LAM_ST*(1-TAU_N_ST)*UPSILON_W_ST*W_C_ST^((1+theta_w)/theta_w)*N_ST /z8_corr_ST +beta*xi_w_ST*(PI_W_ST(+1)/PI_C_ST(+1))^(-1/theta_w)*Z_8_ST(+1); // Equation 49

[name='Foreign aggregate real wage']
W_C_ST^(-1/theta_w)=(1-xi_w_ST)*W_TIL_C_ST^(-1/theta_w)+xi_w_ST*(PI_W_ST/PI_C_ST*W_C_ST(-1))^(-1/theta_w); // Equation 50

[name='Foreign wage indexation scheme']
PI_W_ST=PI_W_ST(-1)^(1-nu_ST)*SS_PI_C_ST^(nu_ST*(1-iota_w_ST))*PI_C_ST(-1)^(nu_ST*iota_w_ST); // Equation 51

[name='Foreign wage dispersion for labor disutility aggregation']
W_AMP_U_ST=(1-xi_w_ST)*(W_TIL_C_ST/W_C_ST)^(-(1+theta_w)/theta_w*(1+chi))+xi_w_ST*W_AMP_U_ST(-1)*(PI_W_ST/PI_C_ST*W_C_ST(-1)/W_C_ST)^(-(1+theta_w)/theta_w*(1+chi)); // Equation 52

[name='Link between foreign effective consumption and actual private and govt. consumption']
C_TIL_ST=C_ST+eta_0*G_ST; // Equation 53

[name='Expression for foreign real marginal cost in terms of foreign good prices']
MC_D_ST=1/(1-alpha)*W_C_ST*GAM_CD_ST*(N_ST/k_ST)^(alpha)*(1/Z_ST^(1-alpha)); // Equation 54

[name='Foreign resetting firms optimality condition']
Z_1_ST=Z_2_ST*P_TIL_D_ST-Z_3_ST*P_TIL_D_ST^(1+(1+theta_p)/theta_p*(1+psi_ST)); // Equation 55

[name='Foreign recursive law of motion for Z_1_ST']
Z_1_ST=(1+psi_ST)*(1+theta_p)/(1+psi_ST+psi_ST*theta_p)*VARSIGMA_ST*LAM_ST*Y_D_ST*VARTHETA_ST^((1+theta_p)/theta_p*(1+psi_ST))*MC_D_ST/GAM_CD_ST+beta_ST*xi_p_ST*(PI_P_ST(+1)/PI_D_ST(+1))^(-(1+theta_p)/theta_p*(1+psi_ST))*Z_1_ST(+1); // Equation 56

[name='Foreign recursive law of motion for Z_2_ST']
Z_2_ST=VARSIGMA_ST*LAM_ST*UPSILON_ST*Y_D_ST/GAM_CD_ST*VARTHETA_ST^((1+theta_p)/theta_p*(1+psi_ST))+beta_ST*xi_p_ST*(PI_P_ST(+1)/PI_D_ST(+1))^(-(1+psi_ST+psi_ST*theta_p)/theta_p)*Z_2_ST(+1); // Equation 57

[name='Foreign recursive law of motion for Z_3_ST']
Z_3_ST=(psi_ST*theta_p)/(1+psi_ST+psi_ST*theta_p)*VARSIGMA_ST*LAM_ST*UPSILON_ST*Y_D_ST/GAM_CD_ST+beta_ST*xi_p_ST*PI_P_ST(+1)/PI_D_ST(+1)*Z_3_ST(+1); // Equation 58

[name='Foreign price indexation scheme']
PI_P_ST=SS_PI_D_ST^(1-iota_ST)*PI_D_ST(-1)^iota_ST; // Equation 59

[name='Foreign aggregate production function']
P_AMP_D_ST*Y_D_ST+P_AMP_M*Y_M=k_ST^(alpha)*(Z_ST*N_ST)^(1-alpha); // Equation 60

[name='Law of motion for foreign domestic price dispersion term']
P_AMP_D_ST=VARTHETA_ST^((1+theta_p)/theta_p*(1+psi_ST))/(1+psi_ST)*Z_4_ST^(-(1+theta_p)/theta_p*(1+psi_ST))+psi_ST/(1+psi_ST); // Equation 61

[name='Recursive law of motion for foreign auxiliary terms governing domestic price dispersion']
Z_4_ST^(-(1+theta_p)*(1+psi_ST)/theta_p)=(1-xi_p_ST)*P_TIL_D_ST^(-(1+theta_p)*(1+psi_ST )/theta_p)+xi_p_ST*(PI_P_ST/PI_D_ST*Z_4_ST(-1))^(-(1+theta_p)*(1+psi_ST)/theta_p); // Equation 62

[name='Law of motion for foreign export price dispersion term']
P_AMP_M=VARTHETA_M_ST^((1+theta_p)/theta_p*(1+psi_m_ST))/(1+psi_m_ST)*Z_M_4_ST^(-(1+theta_p)/theta_p*(1+psi_m_ST))+psi_m_ST/(1+psi_m_ST); // Equation 63

[name='Recursive law of motion for foreign auxiliary terms governing export price dispersion']
Z_M_4_ST^(-(1+theta_p)*(1+psi_m_ST)/theta_p)=(1-xi_m_ST)*P_TIL_M^(-(1+theta_p)*(1+psi_m_ST )/theta_p)+xi_m_ST*(PI_PM/PI_M*Z_M_4_ST(-1))^(-(1+theta_p)*(1+psi_m_ST)/theta_p); // Equation 64

[name='Zero profit condition for foreign final goods producers']
VARTHETA_ST=1+psi_ST-psi_ST*Z_5_ST; // Equation 65

[name='Recursive law of motion for foreign auxiliary term']
Z_5_ST=(1-xi_p_ST)*P_TIL_D_ST+xi_p_ST*PI_P_ST/PI_D_ST*Z_5_ST(-1); // Equation 66

[name='Definition of the foreign aggregate price index']
VARTHETA_ST=Z_6_ST; // Equation 67

[name='Recursive law of motion for foreign auxiliary term']
Z_6_ST^(-(1+psi_ST+psi_ST*theta_p)/theta_p)=(1-xi_p_ST)*P_TIL_D_ST^(-(1+psi_ST+psi_ST*theta_p)/theta_p)+xi_p_ST*(PI_P_ST/PI_D_ST*Z_6_ST(-1))^(-(1+psi_ST+psi_ST*theta_p)/theta_p); // Equation 68

[name='Foreign aggregate resource constraint for domestic sales']
Y_D_ST=(1-omega_c_ST)*GAM_CD_ST^((1+rho_c)/rho_c)*C_ST+(1-omega_g_ST)*GAM_GD_ST^((1+rho_g)/rho_g)*G_ST; // Equation 69

[name='Foreign resetting firms optimality condition for exports']
Z_M_1_ST=Z_M_2_ST*P_TIL_M-Z_M_3_ST*P_TIL_M^(1+(1+theta_p)/theta_p*(1+psi_m_ST)); // Equation 70

[name='Foreign recursive law of motion for Z_M_1_ST']
Z_M_1_ST=(1+psi_m_ST)*(1+theta_p)/(1+psi_m_ST+psi_m_ST*theta_p)*VARSIGMA_ST*LAM_ST*Y_M*VARTHETA_M_ST^((1+theta_p)/theta_p*(1+psi_m_ST))*MC_D_ST/GAM_CD_ST+beta_ST*xi_m_ST*(PI_PM(+1)/PI_M(+1))^(-(1+theta_p)/theta_p*(1+psi_m_ST))*Z_M_1_ST(+1); // Equation 71

[name='Foreign recursive law of motion for Z_M_2_ST']
Z_M_2_ST=VARSIGMA_ST*LAM_ST*UPSILON_M*Y_M/GAM_CM/Q*VARTHETA_M_ST^((1+theta_p)/theta_p*(1+psi_m_ST))+beta_ST*xi_m_ST*(PI_PM(+1)/PI_M(+1))^(-(1+psi_m_ST+psi_m_ST*theta_p)/theta_p)*Z_M_2_ST(+1); // Equation 72

[name='Foreign recursive law of motion for Z_M_3_ST']
Z_M_3_ST=(psi_m_ST*theta_p)/(1+psi_m_ST+psi_m_ST*theta_p)*VARSIGMA_ST*LAM_ST*UPSILON_M*Y_M/GAM_CM/Q+beta_ST*xi_m_ST*PI_PM(+1)/PI_M(+1)*Z_M_3_ST(+1); // Equation 73

[name='Foreign price indexation scheme for exports']
PI_PM=SS_PI_M^(1-iota_m_ST)*PI_M(-1)^iota_m_ST; // Equation 74

[name='Zero profit condition for foreign final export goods producers']
VARTHETA_M_ST=1+psi_m_ST-psi_m_ST*Z_M_5_ST; // Equation 75

[name='Recursive law of motion for foreign auxiliary term']
Z_M_5_ST=(1-xi_m_ST)*P_TIL_M+xi_m_ST*PI_PM/PI_M*Z_M_5_ST(-1); // Equation 76

[name='Definition of the foreign aggregate price index for exports']
VARTHETA_M_ST=Z_M_6_ST; // Equation 77

[name='Recursive law of motion for foreign auxiliary term']
Z_M_6_ST^(-(1+psi_m_ST+psi_m_ST*theta_p)/theta_p)=(1-xi_m_ST)*P_TIL_M^(-(1+psi_m_ST+psi_m_ST*theta_p)/theta_p)+xi_m_ST*(PI_PM/PI_M*Z_M_6_ST(-1))^(-(1+psi_m_ST+psi_m_ST*theta_p)/theta_p); // Equation 78

[name='Foreign aggregate resource constraint for exports']
Y_M=zeta/zeta_ST*(omega_c*(GAM_CM)^((1+rho_c)/rho_c)*C+omega_g*(GAM_GM)^((1+rho_g)/rho_g)*G); // Equation 79

[name='Foreign aggregate to domestically produced price ratio']
GAM_CD_ST=((1-omega_c_ST)+omega_c_ST*(GAM_MD_ST)^(-1/rho_c_ST))^(-rho_c_ST); // Equation 80

[name='Foreign government to domestically produced price ratio']
GAM_GD_ST=((1-omega_g_ST)+omega_g_ST*(GAM_MD_ST)^(-1/rho_g_ST))^(-rho_g_ST); // Equation 81

[name='Foreign aggregate to imported price ratio']
GAM_CM=((1-omega_c)*GAM_MD^(1/rho_c)+omega_c)^(-rho_c); // Equation 82

[name='Foreign government to imported price ratio']
GAM_GM=((1-omega_g)*GAM_MD^(1/rho_g)+omega_g)^(-rho_g); // Equation 83

[name='Foreign consumer price inflation']
PI_C_ST=GAM_CD_ST/GAM_CD_ST(-1)*PI_D_ST; // Equation 84

[name='Definition of foreign import price inflation']
GAM_MD_ST/GAM_MD_ST(-1)=PI_M_ST/PI_D_ST; // Equation 85

[name='Foreign imported component of consumption']
M_C_ST = omega_c_ST*(GAM_CM_ST)^((1+rho_c_ST)/rho_c_ST)*C_ST; // Equation 86

[name='Foreign imported component of government spending']
M_G_ST = omega_g_ST*(GAM_GM_ST)^((1+rho_g_ST)/rho_g_ST)*G_ST; // Equation 87

[name='Determination of foreign monetary policy']
I_ST=max(elb_ST,(1-psi_i_ST)*(SS_I_ST+psi_pi_ST*(PI_C_ST-SS_PI_C_ST)+psi_pid_ST*(PI_D_ST-SS_PI_D_ST)+psi_x_ST*(Y_ST/Y_ST_POT-1))+psi_i_ST*I_ST(-1)+E_I_ST); // Equation 88

//-------------------------------------------------------------------------------------------------------------------------)
// Additional Equations
//-------------------------------------------------------------------------------------------------------------------------)
[name='UIP condition']
(1-TAU_F)*I = I_ST*Q(+1)/Q*PI_C(+1)/PI_C_ST(+1) + GAMMA*I*B_F/(SS_Y_D+SS_Y_M_ST); // Equation 89

[name='Domestic bond market clearing']
B_F = -B-B_P+B_M; // Equation 90

[name='Net foreign assets (modified to account for cfm_nonfa)']
B = ((1-omega_f)*I(-1)/PI_D+omega_f*I_ST(-1)/PI_D*PI_C/PI_C_ST*Q/Q(-1))*B(-1) + (1-omega_b)*(IB(-1)-I(-1))/PI_D*B(-1) + (I(-1)/PI_D-I_ST(-1)/PI_D*PI_C/PI_C_ST*Q/Q(-1))*((omega_p-omega_f)*B_P(-1)-(1-omega_f)*B_M(-1)) + (1-cfm_nonfa)*TAU_F(-1)*I(-1)/PI_D*((1-omega_f)*B_F(-1)+(1-omega_p)*B_P(-1)) + Y_D + GAM_CD/GAM_CM_ST*Q*Y_M_ST - GAM_CD*C - GAM_GD*G; // Equation 91

[name='Gabaix-Maggiori Gamma']
GAMMA = gamma_0*var_e^gamma_1; // Equation 92

[name='Nominal retail interest rate']
IB = I + THETA; // Equation 93

[name='Debt limit constraint',mcp = 'BLIM > 0'] 
THETA = 0; // Equation 94

[name='Distance from the debt limit']
BLIM = B + m*Y(+1); // Equation 95

[name='GDP ']
Y=Y_D+Y_M_ST; // Equation 96

[name='Foreign GDP']
Y_ST=Y_D_ST+Y_M; // Equation 97

[name='Average lifetime utility for sigma = 1']
U=VARSIGMA*(log(C_TIL-varkappa*C_TIL(-1)-SS_C_TIL*NU) - chi_0*W_AMP_U*N^(1+chi)/(1+chi)) + beta*U(+1); // Equation 98

[name='Average foreign lifetime utility for sigma = 1']
U_ST=VARSIGMA_ST*(log(C_TIL_ST-varkappa_ST*C_TIL_ST(-1)-SS_C_TIL_ST*NU_ST) - chi_0_ST*W_AMP_U_ST*N_ST^(1+chi)/(1+chi)) + beta_ST*U_ST(+1); // Equation 99

//-------------------------------------------------------------------------------------------------------------------------)
// Flexible Price Block: Home
//-------------------------------------------------------------------------------------------------------------------------)
[name='FP marginal utility definition (optimality condition with respect to consumption)']
LAM_POT=(C_TIL_POT-varkappa*C_TIL_POT(-1)-SS_C_TIL*NU)^(-1/sigma)/(1+TAU_C); // Equation 100

[name='FP Euler equation (optimality condition with respect to domestic bond holdings)']
LAM_POT=beta*VARSIGMA(+1)/VARSIGMA*I_POT/PI_C_POT(+1)*LAM_POT(+1); // Equation 101

[name='FP labor-leisure indifference condition (optimality condition with respect to labor)']
(1-TAU_N)*W_C_POT=(1+theta_w)/(1+tau_w)*chi_0*N_POT^(chi)/LAM_POT; // Equation 102

[name='FP link between effective consumption and actual private and government consumption']
C_TIL_POT=C_POT+eta_0*G; // Equation 103

[name='FP expression for real marginal cost in terms of domestic goods prices']
(1+tau_p)/(1+theta_p)=(1/Z^(1-alpha))*1/(1-alpha)*W_C_POT*GAM_CD_POT*(N_POT/k)^(alpha); // Equation 104

[name='FP aggregate production function']
Y_D_POT+Y_M_ST_POT=k^(alpha)*(Z*N_POT)^(1-alpha); // Equation 105

[name='FP aggregate resource constraint for domestic sales']
Y_D_POT=(1-omega_c)*GAM_CD_POT^((1+rho_c)/rho_c)*C_POT+(1-omega_g)*GAM_GD_POT^((1+rho_g)/rho_g)*G; // Equation 106

[name='FP aggregate resource constraint for exports']
Y_M_ST_POT=zeta_ST/zeta*(omega_c_ST*(GAM_CM_ST_POT)^((1+rho_c_ST)/rho_c_ST)*C_ST_POT+omega_g_ST*(GAM_GM_ST_POT)^((1+rho_g_ST)/rho_g_ST)*G_ST); // Equation 107

[name='FP aggregate to domestically produced price ratio']
GAM_CD_POT=((1-omega_c)+omega_c*(GAM_MD_POT)^(-1/rho_c))^(-rho_c); // Equation 108

[name='FP government to domestically produced price ratio']
GAM_GD_POT=((1-omega_g)+omega_g*(GAM_MD_POT)^(-1/rho_g))^(-rho_g); // Equation 109

[name='FP foreign aggregate to imported price ratio']
GAM_CM_ST_POT=((1-omega_c_ST)*GAM_MD_POT^(-1/rho_c_ST)+omega_c_ST)^(-rho_c_ST); // Equation 110

[name='FP foreign government to imported price ratio']
GAM_GM_ST_POT=((1-omega_g_ST)*GAM_MD_POT^(-1/rho_g_ST)+omega_g_ST)^(-rho_g_ST); // Equation 111

[name='FP domestic consumer price inflation']
PI_C_POT=GAM_CD_POT/GAM_CD_POT(-1)*PI_D_POT; // Equation 112

[name='FP imported component of consumption']
M_C_POT = omega_c*(GAM_CM_POT)^((1+rho_c)/rho_c)*C_POT; // Equation 113

[name='FP imported component of government spending']
M_G_POT = omega_g*(GAM_GM_POT)^((1+rho_g)/rho_g)*G; // Equation 114

//-------------------------------------------------------------------------------------------------------------------------)
// Flexible Price Block: Foreign
//-------------------------------------------------------------------------------------------------------------------------)
[name='FP foreign marginal utility definition (optimality condition with respect to consumption)']
LAM_ST_POT=(C_TIL_ST_POT-varkappa_ST*C_TIL_ST_POT(-1)-SS_C_TIL_ST*NU_ST)^(-1/sigma)/(1+TAU_C_ST); // Equation 115

[name='FP foreign Euler equation (optimality condition with respect to domestic bond holdings)']
LAM_ST_POT=beta_ST*VARSIGMA_ST(+1)/VARSIGMA_ST*I_ST_POT/PI_C_ST_POT(+1)*LAM_ST_POT(+1); // Equation 116

[name='FP foreign labor-leisure indifference condition (optimality condition with respect to labor)']
(1-TAU_N_ST)*W_C_ST_POT=(1+theta_w)/(1+tau_w)*chi_0_ST*N_ST_POT^(chi)/LAM_ST_POT; // Equation 117

[name='FP link between foreign effective consumption and actual private and govt. cons.']
C_TIL_ST_POT=C_ST_POT+eta_0*G_ST; // Equation 118

[name='FP expression for foreign real marginal cost in terms of foreign good prices']
(1+tau_p)/(1+theta_p)=(1/(Z_ST)^(1-alpha))*1/(1-alpha)*W_C_ST_POT*GAM_CD_ST_POT*(N_ST_POT/k_ST)^(alpha); // Equation 119

[name='FP foreign aggregate production function']
Y_D_ST_POT+Y_M_POT=k_ST^(alpha)*(Z_ST*N_ST_POT)^(1-alpha); // Equation 120

[name='FP foreign aggregate resource constraint for domestic production']
Y_D_ST_POT=(1-omega_c_ST)*GAM_CD_ST_POT^((1+rho_c)/rho_c)*C_ST_POT+(1-omega_g_ST)*GAM_GD_ST_POT^((1+rho_g)/rho_g)*G_ST; // Equation 121

[name='FP foreign aggregate resource constraint for exports']
Y_M_POT=zeta/zeta_ST*(omega_c*(GAM_CM_POT)^((1+rho_c)/rho_c)*C_POT+omega_g*(GAM_GM_POT)^((1+rho_g)/rho_g)*G); // Equation 122

[name='FP foreign aggregate to domestically produced price ratio']
GAM_CD_ST_POT=((1-omega_c_ST)+omega_c_ST*(GAM_MD_POT)^(1/rho_c_ST))^(-rho_c_ST); // Equation 123

[name='FP foreign government to domestically produced price ratio']
GAM_GD_ST_POT=((1-omega_g_ST)+omega_g_ST*(GAM_MD_POT)^(1/rho_g_ST))^(-rho_g_ST); // Equation 124

[name='FP foreign aggregate to imported price ratio']
GAM_CM_POT=((1-omega_c)*GAM_MD_POT^(1/rho_c)+omega_c)^(-rho_c); // Equation 125

[name='FP foreign government to imported price ratio']
GAM_GM_POT=((1-omega_g)*GAM_MD_POT^(1/rho_g)+omega_g)^(-rho_g); // Equation 126

[name='FP foreign consumer price inflation']
PI_C_ST_POT=GAM_CD_ST_POT/GAM_CD_ST_POT(-1)*PI_D_ST_POT; // Equation 127

[name='FP foreign imported component of consumption']
M_C_ST_POT = omega_c_ST*(GAM_CM_ST_POT)^((1+rho_c_ST)/rho_c_ST)*C_ST_POT; // Equation 128

[name='FP foreign imported component of government spending']
M_G_ST_POT = omega_g_ST*(GAM_GM_ST_POT)^((1+rho_g_ST)/rho_g_ST)*G_ST; // Equation 129

//-------------------------------------------------------------------------------------------------------------------------)
// Flexible Price Block: Additional Equations
//-------------------------------------------------------------------------------------------------------------------------)
[name='FP UIP condition']
(1-SS_TAU_F)*I_POT = I_ST_POT*Q_POT(+1)/Q_POT*PI_C_POT(+1)/PI_C_ST_POT(+1) + GAMMA_POT*I_POT*B_F_POT/(SS_Y_D+SS_Y_M_ST); // Equation 130

[name='FP intermediated funds']
B_F_POT = -B_POT-SS_B_P+SS_B_M; // Equation 131

[name='FP net foreign assets (modified to account for cfm_nonfa)']
B_POT = ((1-omega_f)*I_POT(-1)/PI_D_POT+omega_f*I_ST_POT(-1)/PI_D_POT*PI_C_POT/PI_C_ST_POT*Q_POT/Q_POT(-1))*B_POT(-1) + (I_POT(-1)/PI_D_POT-I_ST_POT(-1)/PI_D_POT*PI_C_POT/PI_C_ST_POT*Q_POT/Q_POT(-1))*((omega_f-omega_p)*SS_B_P-(1-omega_f)*SS_B_M) + (1-cfm_nonfa)*SS_TAU_F*I_POT(-1)/PI_D_POT*((1-omega_f)*B_F_POT(-1)+(1-omega_p)*SS_B_P) + Y_D_POT+GAM_CD_POT/GAM_CM_ST_POT*Q_POT*Y_M_ST_POT - GAM_CD_POT*C_POT - GAM_GD_POT*G; // Equation 132

[name='FP Gabaix-Maggiori Gamma']
GAMMA_POT = gamma_0*var_e^gamma_1; // Equation 133

[name='FP real exchange rate']
GAM_MD_POT=Q_POT*GAM_CD_POT/GAM_CD_ST_POT; // Equation 134

[name='FP determination of home monetary policy']
I_POT=(1-psi_i)*(SS_I+psi_pi*(PI_C_POT-SS_PI_C)+psi_pid*(PI_D_POT-SS_PI_D))+psi_i*I_POT(-1); // Equation 135

[name='FP determination of foreign monetary policy']
I_ST_POT=(1-psi_i_ST)*(SS_I_ST+psi_pi_ST*(PI_C_ST_POT-SS_PI_C_ST)+psi_pid_ST*(PI_D_ST_POT-SS_PI_D_ST))+psi_i_ST*I_ST_POT(-1); // Equation 136
 
[name='FP GDP']
Y_POT=Y_D_POT+Y_M_ST_POT; // Equation 137

[name='FP foreign GDP']
Y_ST_POT=Y_D_ST_POT+Y_M_POT; // Equation 138

//-------------------------------------------------------------------------------------------------------------------------)
// Exogenous Processes: Home
//-------------------------------------------------------------------------------------------------------------------------)
[name='Law of motion for preference shock (home)']
VARSIGMA-SS_VARSIGMA=rho_varsigma*(VARSIGMA(-1)-SS_VARSIGMA)+EPS_VARSIGMA+spill_varsigma*EPS_VARSIGMA_ST; // Equation 139

[name='Law of motion for demand shock (home)']
NU-SS_NU=rho_nu*(NU(-1)-SS_NU)+EPS_NU; // Equation 140

[name='Law of motion for consumption taxes (home)']
TAU_C-SS_TAU_C=rho_tau_c*(TAU_C(-1)-SS_TAU_C)+EPS_TAU_C; // Equation 141

[name='Law of motion for labor taxes (home)']
TAU_N-SS_TAU_N=rho_tau_n*(TAU_N(-1)-SS_TAU_N)+EPS_TAU_N; // Equation 142

[name='Law of motion for government consumption (home)']
(G-SS_G)/SS_G=varrho_g*(G(-1)-SS_G)/SS_G+(1/s_gy)*EPS_G; // Equation 143

[name='Law of motion for aggregate productivity (home)']
(Z-SS_Z)/SS_Z=rho_z*(Z(-1)-SS_Z)/SS_Z+EPS_Z+spill_z*EPS_Z_ST; // Equation 144

[name='Law of motion for markup on domestic sales (home)']
ln((1+tau_p)/UPSILON)=rho_upsilon*ln((1+tau_p)/UPSILON(-1))+EPS_UPSILON; // Equation 145

[name='Law of motion for markup on imports (home)']
ln((1+tau_p)/UPSILON_M)=rho_upsilon_m*ln((1+tau_p)/UPSILON_M(-1))+EPS_UPSILON_M; // Equation 146

[name='Law of motion for wage markup (home)']
ln((1+tau_w)/UPSILON_W)=rho_upsilon_w*ln((1+tau_w)/UPSILON_W(-1))+EPS_UPSILON_W+spill_upsilon_w*EPS_UPSILON_W_ST; // Equation 147

[name='Law of motion for monetary policy shock (home)']
E_I=rho_e_i*E_I(-1)+EPS_I+spill_i*EPS_I_ST; // Equation 148

//-------------------------------------------------------------------------------------------------------------------------)
// Exogenous Processes: Foreign
//-------------------------------------------------------------------------------------------------------------------------)
[name='Law of motion for preference shock (foreign)']
VARSIGMA_ST-SS_VARSIGMA_ST=rho_varsigma_ST*(VARSIGMA_ST(-1)-SS_VARSIGMA_ST)+EPS_VARSIGMA_ST; // Equation 149

[name='Law of motion for demand shock (foreign)']
NU_ST-SS_NU_ST=rho_nu_ST*(NU_ST(-1)-SS_NU_ST)+EPS_NU_ST; // Equation 150

[name='Law of motion for consumption taxes (foreign)']
TAU_C_ST-SS_TAU_C_ST=rho_tau_c_ST*(TAU_C_ST(-1)-SS_TAU_C_ST)+EPS_TAU_C_ST; // Equation 151

[name='Law of motion for labor taxes (foreign)']
TAU_N_ST-SS_TAU_N_ST=rho_tau_n_ST*(TAU_N_ST(-1)-SS_TAU_N_ST)+EPS_TAU_N_ST; // Equation 152

[name='Law of motion for government consumption (foreign)']
(G_ST-SS_G_ST)/SS_G_ST=varrho_g_ST*(G_ST(-1)-SS_G_ST)/SS_G_ST+(1/s_gy_ST)*EPS_G_ST; // Equation 153

[name='Law of motion for aggregate productivity (foreign)']
(Z_ST-SS_Z_ST)/SS_Z_ST=rho_z_ST*(Z_ST(-1)-SS_Z_ST)/SS_Z_ST+EPS_Z_ST; // Equation 154

[name='Law of motion for markup on domestic sales (foreign)']
ln((1+tau_p)/UPSILON_ST)=rho_upsilon_ST*ln((1+tau_p)/UPSILON_ST(-1))+EPS_UPSILON_ST; // Equation 155

[name='Law of motion for markup on imports (foreign)']
ln((1+tau_p)/UPSILON_M_ST)=rho_upsilon_m_ST*ln((1+tau_p)/UPSILON_M_ST(-1))+EPS_UPSILON_M_ST; // Equation 156

[name='Law of motion for wage markup (foreign)']
ln((1+tau_w)/UPSILON_W_ST)=rho_upsilon_w_ST*ln((1+tau_w)/UPSILON_W_ST(-1))+EPS_UPSILON_W_ST; // Equation 157

[name='Law of motion for monetary policy shock (foreign)']
E_I_ST=rho_e_i_ST*E_I_ST(-1)+EPS_I_ST; // Equation 158

//-------------------------------------------------------------------------------------------------------------------------)
// Exogenous Processes: Additional 
//-------------------------------------------------------------------------------------------------------------------------)
[name='Law of motion for portfolio inflow']
(B_P-SS_B_P)/SS_Y=varrho_p*(B_P(-1)-SS_B_P)/SS_Y+EPS_B_P; // Equation 159

[name='Law of motion for FXI (home)']
//(B_M-SS_B_M)/SS_Y=varrho_m*(B_M(-1)-SS_B_M)/SS_Y+EPS_B_M+; // Equation 160
(B_M-SS_B_M)/SS_Y=ppsim_bp*(B_P-SS_B_P)/SS_Y-ppsim_theta*THETA+EPS_B_M;

[name='Law of motion for capital inflow taxes (home)']
//TAU_F-SS_TAU_F=rho_tau_f*(TAU_F(-1)-SS_TAU_F)+EPS_TAU_F; // Equation 161
TAU_F-SS_TAU_F=max(0,-ppsif_b*(B-SS_B)/SS_Y)+EPS_TAU_F;

//-------------------------------------------------------------------------------------------------------------------------)
// Variables for calibration
//-------------------------------------------------------------------------------------------------------------------------)
[name='Four quarter growth rate of the real exchange rate']
D4Q_CAL      = 100*log(Q/Q(-4)); // Equation 162

[name='Four quarter wage rate growth']
D4W_CAL      = 100*log(W_C/W_C(-4)); // Equation 163

[name='Four quarter foreign wage growth']
D4W_CAL_ST   = 100*log(W_C_ST/W_C_ST(-4)); // Equation 164

[name='Four quarter output growth']
D4Y_CAL      = 100*log(Y/Y(-4)); // Equation 165

[name='Four quarter foreign output growth']
D4Y_CAL_ST   = 100*log(Y_ST/Y_ST(-4)); // Equation 166

[name='Investment deviation from SS']
I_CAL        = 400*(I-SS_I); // Equation 167

[name='Foreign investment deviation from SS']
I_CAL_ST     = 400*(I_ST-SS_I_ST); // Equation 168

[name='Producer price inflation deviation from SS']
PI_CAL       = 400*(PI_D-SS_PI_D); // Equation 169

[name='Foreign producer price inflation deviation from SS']
PI_CAL_ST    = 400*(PI_C_ST-SS_PI_C_ST); // Equation 170

[name='Deviation of the real exchange rate from SS']
Q_CAL        = 100*log(Q/SS_Q); // Equation 171

[name='Trade balance deviation from SS']
TB_CAL       = 100*(Y_D + GAM_CD/GAM_CM_ST*Q*Y_M_ST - GAM_CD*C - GAM_GD*G) / (Y_D + GAM_CD/GAM_CM_ST*Q*Y_M_ST) - 100 + 100*(SS_C+SS_G)/SS_Y; // Equation 172

[name='UIP deviation from SS']
UIP_CAL      = 100*(GAMMA*I*B_F-SS_GAMMA*SS_I*SS_B_F)/(SS_Y_D+SS_Y_M_ST); // Equation 173

[name='Real wage deviation from SS']
W_CAL        = 100*log(W_C/SS_W_C); // Equation 174

[name='Foreign real wage deviation from SS']
W_CAL_ST     = 100*log(W_C_ST/SS_W_C_ST); // Equation 175

[name='Deviation of output from SS']
Y_CAL        = 100*log(Y/SS_Y); // Equation 176

[name='Deviation of foreign output from SS']
Y_CAL_ST     = 100*log(Y_ST/SS_Y_ST); // Equation 177
end;

//-------------------------------------------------------------------------------------------------------------------------)
// 4. Define the steady state model file
//-------------------------------------------------------------------------------------------------------------------------)

steady_state_model;
//-------------------------------------------------------------------------------------------------------------------------)
// Core Variables
//-------------------------------------------------------------------------------------------------------------------------)
B                   = SS_B;                       //  1.
B_F                 = SS_B_F;                     //  2.
B_M                 = SS_B_M;                     //  3.
B_P                 = SS_B_P;                     //  4.
BLIM                = SS_BLIM;                    //  5.
C                   = SS_C;                       //  6.
C_ST                = SS_C_ST;                    //  7.
C_TIL               = SS_C_TIL;                   //  8.
C_TIL_ST            = SS_C_TIL_ST;                //  9.
E_I                 = SS_E_I;                     // 10.
E_I_ST              = SS_E_I_ST;                  // 11.
G                   = SS_G;                       // 12.
G_ST                = SS_G_ST;                    // 13.
GAM_CD              = SS_GAM_CD;                  // 14.
GAM_CD_ST           = SS_GAM_CD_ST;               // 15.
GAM_CM              = SS_GAM_CM;                  // 16.
GAM_CM_ST           = SS_GAM_CM_ST;               // 17.
GAM_GD              = SS_GAM_GD;                  // 18.
GAM_GD_ST           = SS_GAM_GD_ST;               // 19.
GAM_GM              = SS_GAM_GM;                  // 20.
GAM_GM_ST           = SS_GAM_GM_ST;               // 21.
GAM_MD              = SS_GAM_MD;                  // 22.
GAM_MD_ST           = SS_GAM_MD_ST;               // 23.
GAMMA               = SS_GAMMA;                   // 24.
I                   = SS_I;                       // 25.
I_ST                = SS_I_ST;                    // 26.
IB                  = SS_IB;                      // 27.
LAM                 = SS_LAM;                     // 28.
LAM_ST              = SS_LAM_ST;                  // 29.
M_C                 = SS_M_C;                     // 30.
M_C_ST              = SS_M_C_ST;                  // 31.
M_G                 = SS_M_G;                     // 32.
M_G_ST              = SS_M_G_ST;                  // 33.
MC_D                = SS_MC_D;                    // 34.
MC_D_ST             = SS_MC_D_ST;                 // 35.
N                   = SS_N;                       // 36.
N_ST                = SS_N_ST;                    // 37.
NU                  = SS_NU;                      // 38.
NU_ST               = SS_NU_ST;                   // 39.
P_AMP_D             = SS_P_AMP_D;                 // 40.
P_AMP_D_ST          = SS_P_AMP_D_ST;              // 41.
P_AMP_M             = SS_P_AMP_M;                 // 42.
P_AMP_M_ST          = SS_P_AMP_M_ST;              // 43.
P_TIL_D             = SS_P_TIL_D;                 // 44.
P_TIL_D_ST          = SS_P_TIL_D_ST;              // 45.
P_TIL_M             = SS_P_TIL_M;                 // 46.
P_TIL_M_ST          = SS_P_TIL_M_ST;              // 47.
PI_C                = SS_PI_C;                    // 48.
PI_C_ST             = SS_PI_C_ST;                 // 49.
PI_D                = SS_PI_D;                    // 50.
PI_D_ST             = SS_PI_D_ST;                 // 51.
PI_M                = SS_PI_M;                    // 52.
PI_M_ST             = SS_PI_M_ST;                 // 53.
PI_P                = SS_PI_P;                    // 54.
PI_P_ST             = SS_PI_P_ST;                 // 55.
PI_PM               = SS_PI_PM;                   // 56.
PI_PM_ST            = SS_PI_PM_ST;                // 57.
PI_W                = SS_PI_W;                    // 58.
PI_W_ST             = SS_PI_W_ST;                 // 59.
Q                   = SS_Q;                       // 60.
TAU_C               = SS_TAU_C;                   // 61.
TAU_C_ST            = SS_TAU_C_ST;                // 62.
TAU_F               = SS_TAU_F;                   // 63.
TAU_N               = SS_TAU_N;                   // 64.
TAU_N_ST            = SS_TAU_N_ST;                // 65.
THETA               = SS_THETA;                   // 66.
U                   = SS_U;                       // 67.
U_ST                = SS_U_ST;                    // 68.
UPSILON             = SS_UPSILON;                 // 69.
UPSILON_ST          = SS_UPSILON_ST;              // 70.
UPSILON_M_ST        = SS_UPSILON_M_ST;            // 71.
UPSILON_M           = SS_UPSILON_M;               // 72.
UPSILON_W           = SS_UPSILON_W;               // 73.
UPSILON_W_ST        = SS_UPSILON_W_ST;            // 74.
VARSIGMA            = SS_VARSIGMA;                // 75.
VARSIGMA_ST         = SS_VARSIGMA_ST;             // 76.
VARTHETA            = SS_VARTHETA;                // 77.
VARTHETA_ST         = SS_VARTHETA_ST;             // 78.
VARTHETA_M          = SS_VARTHETA_M;              // 79.
VARTHETA_M_ST       = SS_VARTHETA_M_ST;           // 80.
W_AMP_U             = SS_W_AMP_U;                 // 81.
W_AMP_U_ST          = SS_W_AMP_U_ST;              // 82.
W_C                 = SS_W_C;                     // 83.
W_C_ST              = SS_W_C_ST;                  // 84.
W_TIL_C             = SS_W_TIL_C;                 // 85.
W_TIL_C_ST          = SS_W_TIL_C_ST;              // 86.
Y                   = SS_Y;                       // 87.
Y_ST                = SS_Y_ST;                    // 88.
Y_D                 = SS_Y_D;                     // 89.
Y_D_ST              = SS_Y_D_ST;                  // 90.
Y_M                 = SS_Y_M;                     // 91.
Y_M_ST              = SS_Y_M_ST;                  // 92.
Z                   = SS_Z;                       // 93.
Z_ST                = SS_Z_ST;                    // 94.
Z_1                 = SS_Z_1;                     // 95.
Z_1_ST              = SS_Z_1_ST;                  // 96.
Z_2                 = SS_Z_2;                     // 97.
Z_2_ST              = SS_Z_2_ST;                  // 98.
Z_3                 = SS_Z_3;                     // 99.
Z_3_ST              = SS_Z_3_ST;                  //100.
Z_4                 = SS_Z_4;                     //101.
Z_4_ST              = SS_Z_4_ST;                  //102.
Z_5                 = SS_Z_5;                     //103.
Z_5_ST              = SS_Z_5_ST;                  //104.
Z_6                 = SS_Z_6;                     //105.
Z_6_ST              = SS_Z_6_ST;                  //106.
Z_7                 = SS_Z_7;                     //107.
Z_7_ST              = SS_Z_7_ST;                  //108.
Z_8                 = SS_Z_8;                     //109.
Z_8_ST              = SS_Z_8_ST;                  //110.
Z_M_1               = SS_Z_M_1;                   //111.
Z_M_1_ST            = SS_Z_M_1_ST;                //112.
Z_M_2               = SS_Z_M_2;                   //113.
Z_M_2_ST            = SS_Z_M_2_ST;                //114.
Z_M_3               = SS_Z_M_3;                   //115.
Z_M_3_ST            = SS_Z_M_3_ST;                //116.
Z_M_4               = SS_Z_M_4;                   //117.
Z_M_4_ST            = SS_Z_M_4_ST;                //118.
Z_M_5               = SS_Z_M_5;                   //119.
Z_M_5_ST            = SS_Z_M_5_ST;                //120.
Z_M_6               = SS_Z_M_6;                   //121.
Z_M_6_ST            = SS_Z_M_6_ST;                //122.
//-------------------------------------------------------------------------------------------------------------------------)
// Potential Variables
//-------------------------------------------------------------------------------------------------------------------------)
B_POT               = B;                          //123. 
B_F_POT             = B_F;                        //124. 
C_POT               = C;                          //125. 
C_ST_POT            = C_ST;                       //126. 
C_TIL_POT           = C_TIL;                      //127. 
C_TIL_ST_POT        = C_TIL_ST;                   //128. 
GAM_CD_POT          = GAM_CD;                     //129. 
GAM_CD_ST_POT       = GAM_CD_ST;                  //130. 
GAM_CM_POT          = GAM_CM;                     //131. 
GAM_CM_ST_POT       = GAM_CM_ST;                  //132. 
GAM_GD_POT          = GAM_GD;                     //133. 
GAM_GD_ST_POT       = GAM_GD_ST;                  //134. 
GAM_GM_POT          = GAM_GM;                     //135. 
GAM_GM_ST_POT       = GAM_GM_ST;                  //136. 
GAM_MD_POT          = GAM_MD;                     //137. 
GAMMA_POT           = GAMMA;                      //138. 
I_POT               = I;                          //139.
I_ST_POT            = I_ST;                       //140.
LAM_POT             = LAM;                        //141. 
LAM_ST_POT          = LAM_ST;                     //142. 
M_C_POT             = M_C;                        //143. 
M_C_ST_POT          = M_C_ST;                     //144. 
M_G_POT             = M_G;                        //145. 
M_G_ST_POT          = M_G_ST;                     //146. 
N_POT               = N;                          //147. 
N_ST_POT            = N_ST;                       //148.
PI_C_POT            = PI_C;                       //149.  
PI_C_ST_POT         = PI_C_ST;                    //150.
PI_D_POT            = PI_D;                       //151.
PI_D_ST_POT         = PI_D_ST;                    //152.
Q_POT               = Q;                          //153.
W_C_POT             = W_C;                        //154.
W_C_ST_POT          = W_C_ST;                     //155.
Y_POT               = Y;                          //156.
Y_ST_POT            = Y_ST;                       //157.
Y_D_POT             = Y_D;                        //158.
Y_D_ST_POT          = Y_D_ST;                     //159.
Y_M_POT             = Y_M;                        //160.        
Y_M_ST_POT          = Y_M_ST;                     //161.                            
//-------------------------------------------------------------------------------------------------------------------------)
// Calibration Variables
//-------------------------------------------------------------------------------------------------------------------------)
D4Q_CAL             = 0;                          //162. 
D4W_CAL             = 0;                          //163. 
D4W_CAL_ST          = 0;                          //164. 
D4Y_CAL             = 0;                          //165. 
D4Y_CAL_ST          = 0;                          //166. 
I_CAL               = 0;                          //167. 
I_CAL_ST            = 0;                          //168. 
PI_CAL              = 0;                          //169. 
PI_CAL_ST           = 0;                          //170. 
Q_CAL               = 0;                          //171. 
TB_CAL              = 100*(Y_D + GAM_CD/GAM_CM_ST*Q*Y_M_ST - GAM_CD*C - GAM_GD*G) / (Y_D + GAM_CD/GAM_CM_ST*Q*Y_M_ST) - 100 + 100*(SS_C+SS_G)/SS_Y;   //172.    
UIP_CAL             = 100*(GAMMA*I*B_F-SS_GAMMA*SS_I*SS_B_F)/(SS_Y_D+SS_Y_M_ST);                                                                      //173.    
W_CAL               = 0;                          //174.
W_CAL_ST            = 0;                          //175. 
Y_CAL               = 0;                          //176. 
Y_CAL_ST            = 0;                          //177. 
end;
                                
                                    
//---------------------------------------------------------------------
// 5. Shock Properties and Simulation Instructions
//---------------------------------------------------------------------
steady;
check;

shocks;
//var EPS_G               = (0.0275*s_gy)^2;
//var EPS_G_ST            = (0.0068*s_gy_ST)^2;       
//var EPS_UPSILON         = 0^2;          
//var EPS_UPSILON_ST      = 0^2;     
//var EPS_UPSILON_M       = 0^2;          
//var EPS_UPSILON_M_ST    = 0^2;  
var EPS_UPSILON_W         = 0^2;          
var EPS_UPSILON_W_ST      = 0^2;  
//var EPS_I               = 0^2;          
//var EPS_I_ST            = 0^2; 

var EPS_VARSIGMA_ST = 0.027621^2;
var EPS_Z_ST        = 0.035034^2;

var EPS_VARSIGMA   = 0.02971^2;
var EPS_Z          = 0.0010004^2;
var EPS_B_P        = 0.13022^2;
end;

stoch_simul(order=1,irf=0,nofunctions) Y_CAL D4Y_CAL PI_CAL I_CAL W_CAL D4W_CAL D4Q_CAL TB_CAL UIP_CAL Y_CAL_ST D4Y_CAL_ST PI_CAL_ST I_CAL_ST W_CAL_ST D4W_CAL_ST;


//********************************************************
//************* Non-linear simulations *******************
//********************************************************

/*

options_.ep.stack_solve_algo=0;
options_.ep.solve_algo=4;
options_.ep.maxit = 20;
options_.ep.shk_overwrite = 1;

extended_path(periods=10000,solver_periods=150);

if sim_mode == 1
    %1: Baseline
    simfname = 'SimData_Stress_mponly';
elseif sim_mode == 2 
    %2: FXI responding to spread
    simfname = 'SimData_Stress_fxi_theta1';
elseif sim_mode == 3 
    %3: FXI responding to B_P shocks
    simfname = 'SimData_Stress_fxi_bp05';
elseif sim_mode == 4
    %4: CFM responding to debt
    simfname = 'SimData_Stress_cfm_b001';
elseif sim_mode == 5
    %FXI responding to B_P shocks and CFM responding to debt
    simfname = 'SimData_Stress_fxi_bp05_cfm_b001';
elseif sim_mode == 6
    %6: No sudden stops
    simfname = 'SimData_Stress_noss';

elseif sim_mode == 11
    %11: AE FX markets
    simfname = 'SimData_Stress_deeperFX';
elseif sim_mode == 12
    %12: AE Nominal Rigidities
    simfname = 'SimData_Stress_AEnomrig';
elseif sim_mode == 13
    %13: AE FX Risk Exposure
    simfname = 'SimData_Stress_noFXrisk';
elseif sim_mode == 14
    %14: AE Nominal Rigidities, Kimball still present (Calvo's recomputed)
    simfname = 'SimData_Stress_AEnomrigKimb';
elseif sim_mode == 15
    %15: AE Calibration, except beta
    simfname = 'SimData_Stress_AEcalib';
elseif sim_mode == 16
    %16: AE Nominal Rigidities, Kimball still present in import prices (Calvo's recomputed)
    simfname = 'SimData_Stress_AEnomrigKimbIm';
else
end

Sim_endo = oo_.endo_simul;
Sim_exo = oo_.exo_simul;
Sim_rescale = oo_.rescaling_factors;
Sim_crunch = oo_.crunch_dummy;
save(simfname,'Sim_endo','Sim_exo','Sim_rescale','Sim_crunch','M_');

for i=1:size(Sim_endo,1)
    eval([M_.endo_names{i} '=Sim_endo(' num2str(i) ',:);']);
end

*/

