import TopExamples.grid

#    "MC16a_TOPQ1_FCNC",
TopExamples.grid.Add("MC16a_TOPQ1_FCNC").datasets = [
#MC16a
'mc16_13TeV.410981.aMcAtNloPythia8EvtGen_NNPDF30ME_A14_Q2cgambWpln.deriv.DAOD_TOPQ1.e6908_e5984_a875_r9364_r9315_p3629',
'mc16_13TeV.410980.aMcAtNloPythia8EvtGen_NNPDF30ME_A14_Q2cgambWmln.deriv.DAOD_TOPQ1.e6908_e5984_a875_r9364_r9315_p3629',
'mc16_13TeV.410985.aMcAtNloPythia8EvtGen_NNPDF30ME_A14_Q2ugambWpln.deriv.DAOD_TOPQ1.e6908_e5984_a875_r9364_r9315_p3629',
'mc16_13TeV.410984.aMcAtNloPythia8EvtGen_NNPDF30ME_A14_Q2ugambWmln.deriv.DAOD_TOPQ1.e6908_e5984_a875_r9364_r9315_p3629',

]

#    "MC16a_TOPQ1_FCNC_prod",
TopExamples.grid.Add("MC16a_TOPQ1_FCNC_prod").datasets = [
#MC16a
#ugam
'mc16_13TeV.410650.aMcAtNloPythia8EvtGen_NNPDF30ME_A14_ug2gamt2gambWlnLH.deriv.DAOD_TOPQ1.e6054_e5984_a875_r9364_r9315_p3629',
'mc16_13TeV.410651.aMcAtNloPythia8EvtGen_NNPDF30ME_A14_ug2gamt2gambWlnRH.deriv.DAOD_TOPQ1.e6054_e5984_a875_r9364_r9315_p3629',
#cgam
'mc16_13TeV.410652.aMcAtNloPythia8EvtGen_NNPDF30ME_A14_cg2gamt2gambWlnLH.deriv.DAOD_TOPQ1.e6054_e5984_a875_r9364_r9315_p3629',
'mc16_13TeV.410653.aMcAtNloPythia8EvtGen_NNPDF30ME_A14_cg2gamt2gambWlnRH.deriv.DAOD_TOPQ1.e6054_e5984_a875_r9364_r9315_p3629',
]
#"MC16a_TOPQ1_ttbar_PowPy8",
TopExamples.grid.Add("MC16a_TOPQ1_ttbar_PowPy8").datasets = [
#MC16a
'mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_TOPQ1.e6337_s3126_r9364_p3832',
]

#"MC16a_TOPQ1_ttbar_Var",
TopExamples.grid.Add("MC16a_TOPQ1_ttbar_Var").datasets = [
#MC16a
'mc16_13TeV.410480.PhPy8EG_A14_ttbar_hdamp517p5_SingleLep.deriv.DAOD_TOPQ1.e6454_e5984_a875_r9364_r9315_p3832',
'mc16_13TeV.410482.PhPy8EG_A14_ttbar_hdamp517p5_dil.deriv.DAOD_TOPQ1.e6454_e5984_a875_r9364_r9315_p3832',
'mc16_13TeV.410557.PowhegHerwig7EvtGen_H7UE_tt_hdamp258p75_704_SingleLep.deriv.DAOD_TOPQ1.e6366_e5984_a875_r9364_r9315_p3832',
'mc16_13TeV.410558.PowhegHerwig7EvtGen_H7UE_tt_hdamp258p75_704_dil.deriv.DAOD_TOPQ1.e6366_e5984_a875_r9364_r9315_p3832',
'mc16_13TeV.410464.aMcAtNloPy8EvtGen_MEN30NLO_A14N23LO_ttbar_noShWe_SingleLep.deriv.DAOD_TOPQ1.e6762_e5984_a875_r9364_r9315_p3832',
'mc16_13TeV.410465.aMcAtNloPy8EvtGen_MEN30NLO_A14N23LO_ttbar_noShWe_dil.deriv.DAOD_TOPQ1.e6762_e5984_a875_r9364_r9315_p3832',
]

#"MC16a_TOPQ1_ttgamma",
TopExamples.grid.Add("MC16a_TOPQ1_ttgamma").datasets = [
#MC16a
'mc16_13TeV.410389.MadGraphPythia8EvtGen_A14NNPDF23_ttgamma_nonallhadronic.deriv.DAOD_TOPQ1.e6155_s3126_r9364_p3832',
]

#"MC16a_TOPQ1_singleTop",
TopExamples.grid.Add("MC16a_TOPQ1_singleTop").datasets = [
#MC16a
# t-channel
'mc16_13TeV.410658.PhPy8EG_A14_tchan_BW50_lept_top.deriv.DAOD_TOPQ1.e6671_s3126_r9364_p3832',
'mc16_13TeV.410659.PhPy8EG_A14_tchan_BW50_lept_antitop.deriv.DAOD_TOPQ1.e6671_s3126_r9364_p3832',
# s-channel
'mc16_13TeV.410644.PowhegPythia8EvtGen_A14_singletop_schan_lept_top.deriv.DAOD_TOPQ1.e6527_s3126_r9364_p3832',
'mc16_13TeV.410645.PowhegPythia8EvtGen_A14_singletop_schan_lept_antitop.deriv.DAOD_TOPQ1.e6527_s3126_r9364_p3832',
# tW-channel
'mc16_13TeV.410646.PowhegPythia8EvtGen_A14_Wt_DR_inclusive_top.deriv.DAOD_TOPQ1.e6552_s3126_r9364_p3832',
'mc16_13TeV.410647.PowhegPythia8EvtGen_A14_Wt_DR_inclusive_antitop.deriv.DAOD_TOPQ1.e6552_s3126_r9364_p3832',
]

#"MC16a_TOPQ1_singleTop_Var",
TopExamples.grid.Add("MC16a_TOPQ1_singleTop_Var").datasets = [
#MC16a
# t-channel
'mc16_13TeV.411032.PowhegHerwig7EvtGen_H7UE_704_tchan_lept_antitop.deriv.DAOD_TOPQ1.e6719_e5984_a875_r9364_r9315_p3832',
'mc16_13TeV.411033.PowhegHerwig7EvtGen_H7UE_704_tchan_lept_top.deriv.DAOD_TOPQ1.e6719_e5984_a875_r9364_r9315_p3832',
'mc16_13TeV.410560.MadGraphPythia8EvtGen_A14_tZ_4fl_tchan_noAllHad.deriv.DAOD_TOPQ1.e5803_s3126_r9364_r9315_p3832',#MGPyth
# s-channel
'mc16_13TeV.411034.PhHerwig7EG_H7UE_singletop_schan_lept_top.deriv.DAOD_TOPQ1.e6734_e5984_a875_r9364_r9315_p3832',
'mc16_13TeV.411035.PhHerwig7EG_H7UE_singletop_schan_lept_antitop.deriv.DAOD_TOPQ1.e6734_e5984_a875_r9364_r9315_p3832',
# tW-channel
'mc16_13TeV.411036.PowhegHerwig7EvtGen_H7UE_Wt_DR_inclusive_top.deriv.DAOD_TOPQ1.e6702_e5984_a875_r9364_r9315_p3832',
'mc16_13TeV.411037.PowhegHerwig7EvtGen_H7UE_Wt_DR_inclusive_antitop.deriv.DAOD_TOPQ1.e6702_e5984_a875_r9364_r9315_p3832',
'mc16_13TeV.410654.PowhegPythia8EvtGen_A14_Wt_DS_inclusive_top.deriv.DAOD_TOPQ1.e6552_e5984_s3126_r9364_r9315_p3832',
'mc16_13TeV.410655.PowhegPythia8EvtGen_A14_Wt_DS_inclusive_antitop.deriv.DAOD_TOPQ1.e6552_e5984_s3126_r9364_r9315_p3832',
'mc16_13TeV.412002.aMcAtNloPythia8EvtGen_HThalfscale_tW_inclusive.deriv.DAOD_TOPQ1.e6817_e5984_a875_r9364_r9315_p3832',
]

#"MC16a_TOPQ1_ttV",
TopExamples.grid.Add("MC16a_TOPQ1_ttV").datasets = [
#MC16a
'mc16_13TeV.410155.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttW.deriv.DAOD_TOPQ1.e5070_s3126_r9364_p3832',
'mc16_13TeV.410156.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttZnunu.deriv.DAOD_TOPQ1.e5070_s3126_r9364_p3832',
'mc16_13TeV.410157.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttZqq.deriv.DAOD_TOPQ1.e5070_s3126_r9364_p3832',
'mc16_13TeV.410218.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttee.deriv.DAOD_TOPQ1.e5070_s3126_r9364_p3832',
'mc16_13TeV.410219.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttmumu.deriv.DAOD_TOPQ1.e5070_s3126_r9364_p3832',
'mc16_13TeV.410220.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_tttautau.deriv.DAOD_TOPQ1.e5070_s3126_r9364_p3832',
]

#"MC16a_TOPQ1_diboson_Sherpa",
TopExamples.grid.Add("MC16a_TOPQ1_diboson_Sherpa").datasets = [
#MC16a
'mc16_13TeV.364250.Sherpa_222_NNPDF30NNLO_llll.deriv.DAOD_TOPQ1.e5894_s3126_r9364_p3830',
'mc16_13TeV.364253.Sherpa_222_NNPDF30NNLO_lllv.deriv.DAOD_TOPQ1.e5916_s3126_r9364_p3830',
'mc16_13TeV.364254.Sherpa_222_NNPDF30NNLO_llvv.deriv.DAOD_TOPQ1.e5916_s3126_r9364_p3830',
'mc16_13TeV.364255.Sherpa_222_NNPDF30NNLO_lvvv.deriv.DAOD_TOPQ1.e5916_s3126_r9364_p3830',
#Sherpa 221 unfiltered
'mc16_13TeV.363355.Sherpa_221_NNPDF30NNLO_ZqqZvv.deriv.DAOD_TOPQ1.e5525_s3126_r9364_p3830',
'mc16_13TeV.363356.Sherpa_221_NNPDF30NNLO_ZqqZll.deriv.DAOD_TOPQ1.e5525_s3126_r9364_p3830',
'mc16_13TeV.363357.Sherpa_221_NNPDF30NNLO_WqqZvv.deriv.DAOD_TOPQ1.e5525_s3126_r9364_p3830',
'mc16_13TeV.363358.Sherpa_221_NNPDF30NNLO_WqqZll.deriv.DAOD_TOPQ1.e5525_s3126_r9364_p3830',
'mc16_13TeV.363359.Sherpa_221_NNPDF30NNLO_WpqqWmlv.deriv.DAOD_TOPQ1.e5583_s3126_r9364_p3830',
'mc16_13TeV.363360.Sherpa_221_NNPDF30NNLO_WplvWmqq.deriv.DAOD_TOPQ1.e5983_s3126_r9364_p3830',
'mc16_13TeV.363489.Sherpa_221_NNPDF30NNLO_WlvZqq.deriv.DAOD_TOPQ1.e5525_s3126_r9364_p3830',
]

TopExamples.grid.Add("MC16a_TOPQ1_Wjets_Sherpa221").datasets = [
#MC16a
# W -> e nu  
'mc16_13TeV.364170.Sherpa_221_NNPDF30NNLO_Wenu_MAXHTPTV0_70_CVetoBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364171.Sherpa_221_NNPDF30NNLO_Wenu_MAXHTPTV0_70_CFilterBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364172.Sherpa_221_NNPDF30NNLO_Wenu_MAXHTPTV0_70_BFilter.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364173.Sherpa_221_NNPDF30NNLO_Wenu_MAXHTPTV70_140_CVetoBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364174.Sherpa_221_NNPDF30NNLO_Wenu_MAXHTPTV70_140_CFilterBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364175.Sherpa_221_NNPDF30NNLO_Wenu_MAXHTPTV70_140_BFilter.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364176.Sherpa_221_NNPDF30NNLO_Wenu_MAXHTPTV140_280_CVetoBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364177.Sherpa_221_NNPDF30NNLO_Wenu_MAXHTPTV140_280_CFilterBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364178.Sherpa_221_NNPDF30NNLO_Wenu_MAXHTPTV140_280_BFilter.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364179.Sherpa_221_NNPDF30NNLO_Wenu_MAXHTPTV280_500_CVetoBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364180.Sherpa_221_NNPDF30NNLO_Wenu_MAXHTPTV280_500_CFilterBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364181.Sherpa_221_NNPDF30NNLO_Wenu_MAXHTPTV280_500_BFilter.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364182.Sherpa_221_NNPDF30NNLO_Wenu_MAXHTPTV500_1000.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364183.Sherpa_221_NNPDF30NNLO_Wenu_MAXHTPTV1000_E_CMS.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
# W -> mu nu
'mc16_13TeV.364156.Sherpa_221_NNPDF30NNLO_Wmunu_MAXHTPTV0_70_CVetoBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364157.Sherpa_221_NNPDF30NNLO_Wmunu_MAXHTPTV0_70_CFilterBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364158.Sherpa_221_NNPDF30NNLO_Wmunu_MAXHTPTV0_70_BFilter.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364159.Sherpa_221_NNPDF30NNLO_Wmunu_MAXHTPTV70_140_CVetoBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364160.Sherpa_221_NNPDF30NNLO_Wmunu_MAXHTPTV70_140_CFilterBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364161.Sherpa_221_NNPDF30NNLO_Wmunu_MAXHTPTV70_140_BFilter.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364162.Sherpa_221_NNPDF30NNLO_Wmunu_MAXHTPTV140_280_CVetoBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364163.Sherpa_221_NNPDF30NNLO_Wmunu_MAXHTPTV140_280_CFilterBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364164.Sherpa_221_NNPDF30NNLO_Wmunu_MAXHTPTV140_280_BFilter.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364165.Sherpa_221_NNPDF30NNLO_Wmunu_MAXHTPTV280_500_CVetoBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364166.Sherpa_221_NNPDF30NNLO_Wmunu_MAXHTPTV280_500_CFilterBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364167.Sherpa_221_NNPDF30NNLO_Wmunu_MAXHTPTV280_500_BFilter.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364168.Sherpa_221_NNPDF30NNLO_Wmunu_MAXHTPTV500_1000.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364169.Sherpa_221_NNPDF30NNLO_Wmunu_MAXHTPTV1000_E_CMS.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
 # W -> tau nu
'mc16_13TeV.364184.Sherpa_221_NNPDF30NNLO_Wtaunu_MAXHTPTV0_70_CVetoBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364185.Sherpa_221_NNPDF30NNLO_Wtaunu_MAXHTPTV0_70_CFilterBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364186.Sherpa_221_NNPDF30NNLO_Wtaunu_MAXHTPTV0_70_BFilter.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364187.Sherpa_221_NNPDF30NNLO_Wtaunu_MAXHTPTV70_140_CVetoBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364188.Sherpa_221_NNPDF30NNLO_Wtaunu_MAXHTPTV70_140_CFilterBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364189.Sherpa_221_NNPDF30NNLO_Wtaunu_MAXHTPTV70_140_BFilter.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364190.Sherpa_221_NNPDF30NNLO_Wtaunu_MAXHTPTV140_280_CVetoBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364191.Sherpa_221_NNPDF30NNLO_Wtaunu_MAXHTPTV140_280_CFilterBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364192.Sherpa_221_NNPDF30NNLO_Wtaunu_MAXHTPTV140_280_BFilter.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364193.Sherpa_221_NNPDF30NNLO_Wtaunu_MAXHTPTV280_500_CVetoBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364194.Sherpa_221_NNPDF30NNLO_Wtaunu_MAXHTPTV280_500_CFilterBVeto.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364195.Sherpa_221_NNPDF30NNLO_Wtaunu_MAXHTPTV280_500_BFilter.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364196.Sherpa_221_NNPDF30NNLO_Wtaunu_MAXHTPTV500_1000.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
'mc16_13TeV.364197.Sherpa_221_NNPDF30NNLO_Wtaunu_MAXHTPTV1000_E_CMS.deriv.DAOD_TOPQ1.e5340_s3126_r9364_p3830',
]

#"MC16a_TOPQ1_Zjets_Sherpa221",
TopExamples.grid.Add("MC16a_TOPQ1_Zjets_Sherpa221").datasets = [
#MC16a
# Z -> e e 
'mc16_13TeV.364114.Sherpa_221_NNPDF30NNLO_Zee_MAXHTPTV0_70_CVetoBVeto.deriv.DAOD_TOPQ1.e5299_s3126_r9364_p3830',
'mc16_13TeV.364115.Sherpa_221_NNPDF30NNLO_Zee_MAXHTPTV0_70_CFilterBVeto.deriv.DAOD_TOPQ1.e5299_s3126_r9364_p3830',
'mc16_13TeV.364116.Sherpa_221_NNPDF30NNLO_Zee_MAXHTPTV0_70_BFilter.deriv.DAOD_TOPQ1.e5299_s3126_r9364_p3830',
'mc16_13TeV.364117.Sherpa_221_NNPDF30NNLO_Zee_MAXHTPTV70_140_CVetoBVeto.deriv.DAOD_TOPQ1.e5299_s3126_r9364_p3830',
'mc16_13TeV.364118.Sherpa_221_NNPDF30NNLO_Zee_MAXHTPTV70_140_CFilterBVeto.deriv.DAOD_TOPQ1.e5299_s3126_r9364_p3830',
'mc16_13TeV.364119.Sherpa_221_NNPDF30NNLO_Zee_MAXHTPTV70_140_BFilter.deriv.DAOD_TOPQ1.e5299_s3126_r9364_p3830',
'mc16_13TeV.364120.Sherpa_221_NNPDF30NNLO_Zee_MAXHTPTV140_280_CVetoBVeto.deriv.DAOD_TOPQ1.e5299_s3126_r9364_p3830',
'mc16_13TeV.364121.Sherpa_221_NNPDF30NNLO_Zee_MAXHTPTV140_280_CFilterBVeto.deriv.DAOD_TOPQ1.e5299_s3126_r9364_p3830',
'mc16_13TeV.364122.Sherpa_221_NNPDF30NNLO_Zee_MAXHTPTV140_280_BFilter.deriv.DAOD_TOPQ1.e5299_s3126_r9364_p3830',
'mc16_13TeV.364123.Sherpa_221_NNPDF30NNLO_Zee_MAXHTPTV280_500_CVetoBVeto.deriv.DAOD_TOPQ1.e5299_s3126_r9364_p3830',
'mc16_13TeV.364124.Sherpa_221_NNPDF30NNLO_Zee_MAXHTPTV280_500_CFilterBVeto.deriv.DAOD_TOPQ1.e5299_s3126_r9364_p3830',
'mc16_13TeV.364125.Sherpa_221_NNPDF30NNLO_Zee_MAXHTPTV280_500_BFilter.deriv.DAOD_TOPQ1.e5299_s3126_r9364_p3830',
'mc16_13TeV.364126.Sherpa_221_NNPDF30NNLO_Zee_MAXHTPTV500_1000.deriv.DAOD_TOPQ1.e5299_s3126_r9364_p3830',
'mc16_13TeV.364127.Sherpa_221_NNPDF30NNLO_Zee_MAXHTPTV1000_E_CMS.deriv.DAOD_TOPQ1.e5299_s3126_r9364_p3830',
# Z -> mu mu
'mc16_13TeV.364100.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_CVetoBVeto.deriv.DAOD_TOPQ1.e5271_s3126_r9364_p3830',
'mc16_13TeV.364101.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_CFilterBVeto.deriv.DAOD_TOPQ1.e5271_s3126_r9364_p3830',
'mc16_13TeV.364102.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_BFilter.deriv.DAOD_TOPQ1.e5271_s3126_r9364_p3830',
'mc16_13TeV.364103.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV70_140_CVetoBVeto.deriv.DAOD_TOPQ1.e5271_s3126_r9364_p3830',
'mc16_13TeV.364104.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV70_140_CFilterBVeto.deriv.DAOD_TOPQ1.e5271_s3126_r9364_p3830',
'mc16_13TeV.364105.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV70_140_BFilter.deriv.DAOD_TOPQ1.e5271_s3126_r9364_p3830',
'mc16_13TeV.364106.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV140_280_CVetoBVeto.deriv.DAOD_TOPQ1.e5271_s3126_r9364_p3830',
'mc16_13TeV.364107.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV140_280_CFilterBVeto.deriv.DAOD_TOPQ1.e5271_s3126_r9364_p3830',
'mc16_13TeV.364108.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV140_280_BFilter.deriv.DAOD_TOPQ1.e5271_s3126_r9364_p3830',
'mc16_13TeV.364109.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV280_500_CVetoBVeto.deriv.DAOD_TOPQ1.e5271_s3126_r9364_p3830',
'mc16_13TeV.364110.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV280_500_CFilterBVeto.deriv.DAOD_TOPQ1.e5271_s3126_r9364_p3830',
'mc16_13TeV.364111.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV280_500_BFilter.deriv.DAOD_TOPQ1.e5271_s3126_r9364_p3830',
'mc16_13TeV.364112.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV500_1000.deriv.DAOD_TOPQ1.e5271_s3126_r9364_p3830',
'mc16_13TeV.364113.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV1000_E_CMS.deriv.DAOD_TOPQ1.e5271_s3126_r9364_p3830',
# Z -> tau tau
'mc16_13TeV.364128.Sherpa_221_NNPDF30NNLO_Ztautau_MAXHTPTV0_70_CVetoBVeto.deriv.DAOD_TOPQ1.e5307_s3126_r9364_p3830',
'mc16_13TeV.364129.Sherpa_221_NNPDF30NNLO_Ztautau_MAXHTPTV0_70_CFilterBVeto.deriv.DAOD_TOPQ1.e5307_s3126_r9364_p3830',
'mc16_13TeV.364130.Sherpa_221_NNPDF30NNLO_Ztautau_MAXHTPTV0_70_BFilter.deriv.DAOD_TOPQ1.e5307_s3126_r9364_p3830',
'mc16_13TeV.364131.Sherpa_221_NNPDF30NNLO_Ztautau_MAXHTPTV70_140_CVetoBVeto.deriv.DAOD_TOPQ1.e5307_s3126_r9364_p3830',
'mc16_13TeV.364132.Sherpa_221_NNPDF30NNLO_Ztautau_MAXHTPTV70_140_CFilterBVeto.deriv.DAOD_TOPQ1.e5307_s3126_r9364_p3830',
'mc16_13TeV.364133.Sherpa_221_NNPDF30NNLO_Ztautau_MAXHTPTV70_140_BFilter.deriv.DAOD_TOPQ1.e5307_s3126_r9364_p3830',
'mc16_13TeV.364134.Sherpa_221_NNPDF30NNLO_Ztautau_MAXHTPTV140_280_CVetoBVeto.deriv.DAOD_TOPQ1.e5307_s3126_r9364_p3830',
'mc16_13TeV.364135.Sherpa_221_NNPDF30NNLO_Ztautau_MAXHTPTV140_280_CFilterBVeto.deriv.DAOD_TOPQ1.e5307_s3126_r9364_p3830',
'mc16_13TeV.364136.Sherpa_221_NNPDF30NNLO_Ztautau_MAXHTPTV140_280_BFilter.deriv.DAOD_TOPQ1.e5307_s3126_r9364_p3830',
'mc16_13TeV.364137.Sherpa_221_NNPDF30NNLO_Ztautau_MAXHTPTV280_500_CVetoBVeto.deriv.DAOD_TOPQ1.e5307_s3126_r9364_p3830',
'mc16_13TeV.364138.Sherpa_221_NNPDF30NNLO_Ztautau_MAXHTP500_CFilterBVeto.deriv.DAOD_TOPQ1.e5313_s3126_r9364_p3830',
'mc16_13TeV.364139.Sherpa_221_NNPDF30NNLO_Ztautau_MAXHTPTV280_500_BFilter.deriv.DAOD_TOPQ1.e5313_s3126_r9364_p3830',
'mc16_13TeV.364140.Sherpa_221_NNPDF30NNLO_Ztautau_MAXHTPTV500_1000.deriv.DAOD_TOPQ1.e5307_s3126_r9364_p3830',
'mc16_13TeV.364141.Sherpa_221_NNPDF30NNLO_Ztautau_MAXHTPTV1000_E_CMS.deriv.DAOD_TOPQ1.e5307_s3126_r9364_p3830',
]

#"MC16a_TOPQ1_V_photons", #WJets+ZJets + GAMMA
TopExamples.grid.Add("MC16a_TOPQ1_V_photons").datasets = [
#MC16a
# Z+jets+gamma
#MC16a
# Zgamma with Z -> ee
'mc16_13TeV.366140.Sh_224_NN30NNLO_eegamma_LO_pty_7_15.deriv.DAOD_TOPQ1.e7006_s3126_r9364_p3830',
'mc16_13TeV.366141.Sh_224_NN30NNLO_eegamma_LO_pty_15_35.deriv.DAOD_TOPQ1.e7006_s3126_r9364_p3830',
'mc16_13TeV.366142.Sh_224_NN30NNLO_eegamma_LO_pty_35_70.deriv.DAOD_TOPQ1.e7006_s3126_r9364_p3830',
'mc16_13TeV.366143.Sh_224_NN30NNLO_eegamma_LO_pty_70_140.deriv.DAOD_TOPQ1.e7006_s3126_r9364_p3830',
'mc16_13TeV.366144.Sh_224_NN30NNLO_eegamma_LO_pty_140_E_CMS.deriv.DAOD_TOPQ1.e7006_s3126_r9364_p3830',
# Zgamma with Z -> mumu
'mc16_13TeV.366145.Sh_224_NN30NNLO_mumugamma_LO_pty_7_15.deriv.DAOD_TOPQ1.e7006_s3126_r9364_p3830',
'mc16_13TeV.366146.Sh_224_NN30NNLO_mumugamma_LO_pty_15_35.deriv.DAOD_TOPQ1.e7006_s3126_r9364_p3830',
'mc16_13TeV.366147.Sh_224_NN30NNLO_mumugamma_LO_pty_35_70.deriv.DAOD_TOPQ1.e7006_s3126_r9364_p3830',
'mc16_13TeV.366148.Sh_224_NN30NNLO_mumugamma_LO_pty_70_140.deriv.DAOD_TOPQ1.e7006_s3126_r9364_p3830',
'mc16_13TeV.366149.Sh_224_NN30NNLO_mumugamma_LO_pty_140_E_CMS.deriv.DAOD_TOPQ1.e7006_s3126_r9364_p3830',
# Zgamma with Z -> tautau
'mc16_13TeV.366150.Sh_224_NN30NNLO_tautaugamma_LO_pty_7_15.deriv.DAOD_TOPQ1.e7029_s3126_r9364_p3830',
'mc16_13TeV.366151.Sh_224_NN30NNLO_tautaugamma_LO_pty_15_35.deriv.DAOD_TOPQ1.e7029_s3126_r9364_p3830',
'mc16_13TeV.366152.Sh_224_NN30NNLO_tautaugamma_LO_pty_35_70.deriv.DAOD_TOPQ1.e7029_s3126_r9364_p3830',
'mc16_13TeV.366153.Sh_224_NN30NNLO_tautaugamma_LO_pty_70_140.deriv.DAOD_TOPQ1.e7029_s3126_r9364_p3830',
'mc16_13TeV.366154.Sh_224_NN30NNLO_tautaugamma_LO_pty_140_E_CMS.deriv.DAOD_TOPQ1.e7029_s3126_r9364_p3830',
#MC16a
  # W+jets+gamma  
'mc16_13TeV.364521.Sherpa_222_NNPDF30NNLO_enugamma_pty_7_15.deriv.DAOD_TOPQ1.e5928_s3126_r9364_p3830',
'mc16_13TeV.364522.Sherpa_222_NNPDF30NNLO_enugamma_pty_15_35.deriv.DAOD_TOPQ1.e5928_s3126_r9364_p3830',
'mc16_13TeV.364523.Sherpa_222_NNPDF30NNLO_enugamma_pty_35_70.deriv.DAOD_TOPQ1.e5928_s3126_r9364_p3830',
'mc16_13TeV.364524.Sherpa_222_NNPDF30NNLO_enugamma_pty_70_140.deriv.DAOD_TOPQ1.e5928_s3126_r9364_p3830',
'mc16_13TeV.364525.Sherpa_222_NNPDF30NNLO_enugamma_pty_140_E_CMS.deriv.DAOD_TOPQ1.e5928_s3126_r9364_p3830',
'mc16_13TeV.364526.Sherpa_222_NNPDF30NNLO_munugamma_pty_7_15.deriv.DAOD_TOPQ1.e5928_s3126_r9364_p3830',
'mc16_13TeV.364527.Sherpa_222_NNPDF30NNLO_munugamma_pty_15_35.deriv.DAOD_TOPQ1.e5928_s3126_r9364_p3830',
'mc16_13TeV.364528.Sherpa_222_NNPDF30NNLO_munugamma_pty_35_70.deriv.DAOD_TOPQ1.e5928_s3126_r9364_p3830',
'mc16_13TeV.364529.Sherpa_222_NNPDF30NNLO_munugamma_pty_70_140.deriv.DAOD_TOPQ1.e5928_s3126_r9364_p3830',
'mc16_13TeV.364530.Sherpa_222_NNPDF30NNLO_munugamma_pty_140_E_CMS.deriv.DAOD_TOPQ1.e5928_s3126_r9364_p3830',
'mc16_13TeV.364531.Sherpa_222_NNPDF30NNLO_taunugamma_pty_7_15.deriv.DAOD_TOPQ1.e5928_s3126_r9364_p3830',
'mc16_13TeV.364532.Sherpa_222_NNPDF30NNLO_taunugamma_pty_15_35.deriv.DAOD_TOPQ1.e5928_s3126_r9364_p3830',
'mc16_13TeV.364533.Sherpa_222_NNPDF30NNLO_taunugamma_pty_35_70.deriv.DAOD_TOPQ1.e5928_s3126_r9364_p3830',
'mc16_13TeV.364534.Sherpa_222_NNPDF30NNLO_taunugamma_pty_70_140.deriv.DAOD_TOPQ1.e5928_s3126_r9364_p3830',
'mc16_13TeV.364535.Sherpa_222_NNPDF30NNLO_taunugamma_pty_140_E_CMS.deriv.DAOD_TOPQ1.e5928_s3126_r9364_p3830',
]






