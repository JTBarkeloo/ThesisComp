from ROOT import *
gROOT.SetStyle("ATLAS")
channel=['e+jets','#mu+jets','Combined']
files =['FCNC_All_ejets/Limits/asymptotics/myLimit_CL95.root','FCNC_All_mujets/Limits/asymptotics/myLimit_CL95.root','Comb/Limits/asymptotics/myLimit_CL95.root']
#t1.Print()
#print t1.obs_upperlimit
gStyle.SetOptStat(0)
N=len(files) #nfiles
ymin = -0.5
ymax=N-0.5
c=TCanvas('c','c',700,500)
g_obs = TGraphErrors(N)
g_exp = TGraphErrors(N)
g_1s  = TGraphAsymmErrors(N)
g_2s  = TGraphAsymmErrors(N)
Ndiv=N+1
xmax=0
for i in range(N):
	print "Channel: ", channel[i]
	f=TFile(files[i],'read')
	t=f.Get('stats')
	t.GetEntry(0)
	print "Observed: ", t.obs_upperlimit
	print "Expected: ", t.exp_upperlimit
	print "Expected+2: ", t.exp_upperlimit_plus2
	g_obs.SetPoint(N-i-1,t.obs_upperlimit,N-i-1)
	g_exp.SetPoint(N-i-1,t.exp_upperlimit,N-i-1)
	g_1s.SetPoint(N-i-1,t.exp_upperlimit,N-i-1)
	g_2s.SetPoint(N-i-1,t.exp_upperlimit,N-i-1)
	g_obs.SetPointError(N-i-1,0,0.5)
	g_exp.SetPointError(N-i-1,0,0.5)
	g_1s.SetPointError(N-i-1,t.exp_upperlimit-t.exp_upperlimit_minus1,t.exp_upperlimit_plus1-t.exp_upperlimit,0.5,0.5)
	g_2s.SetPointError(N-i-1,t.exp_upperlimit-t.exp_upperlimit_minus2,t.exp_upperlimit_plus2-t.exp_upperlimit,0.5,0.5)
	if t.exp_upperlimit_plus2 > xmax:
		xmax = t.exp_upperlimit_plus2
	if t.obs_upperlimit > xmax:
		xmax=t.obs_upperlimit
#if 0.24>xmax:
xmax=xmax*1.3
g_obs.SetLineWidth(3)
g_exp.SetLineWidth(3)
g_exp.SetLineStyle(2)
g_1s.SetFillColor(kGreen)
g_1s.SetLineWidth(3)
g_1s.SetLineStyle(2)
g_2s.SetFillColor(kYellow)
g_2s.SetLineWidth(3)
g_2s.SetLineWidth(2)

g_2s.SetMarkerSize(0)
g_1s.SetMarkerSize(0)
g_exp.SetMarkerSize(0)
g_obs.SetMarkerSize(0)

dummy = TH1D('','',1,0,xmax)
dummy.Draw()
dummy.SetMinimum(ymin)
dummy.SetMaximum(ymax)
dummy.SetLineColor(kWhite)
dummy.GetYaxis().Set(N,ymin,ymax)
dummy.GetYaxis().SetNdivisions(Ndiv)
for i in range(N):
	dummy.GetYaxis().SetBinLabel(N-i,channel[i])
g_2s.Draw("E2 same")
g_1s.Draw("E2 same")
g_exp.Draw("E same")
g_obs.Draw("E same")

l_SM = TLine(1,-0.5,1,N-0.5)
l_SM.SetLineWidth(2)
l_SM.SetLineColor(kGray)
l_SM.Draw("same")

c.RedrawAxis()
gPad.SetRightMargin(3*gPad.GetRightMargin())
gPad.SetBottomMargin(1.15*gPad.GetBottomMargin())
gPad.SetTopMargin(1.8*gPad.GetTopMargin())
dummy.GetXaxis().SetTitle("95% CL limit on #sigma^{FCNC}/#sigma^{FCNC}_{nominal}")

leg=TLegend(0.65,0.2,0.95,0.4)
leg.SetTextSize(gStyle.GetTextSize())
leg.SetTextFont(gStyle.GetTextFont())
#leg.SetFillStyle(0)
leg.SetBorderSize(0)
leg.AddEntry(g_1s,"Expected #pm 1#sigma","lf")
leg.AddEntry(g_2s,"Expected #pm 2#sigma","lf")
leg.AddEntry(g_obs,"Observed","l")
leg.Draw()

c.SaveAs('LimitPlot.png')




