import sys, os, time, shutil

import foundation as fd
from foundation import models
from foundation import util
from foundation import train as trn

root = os.environ['FOUNDATION_SAVE_DIR']
print(root)

raw = '''AE	Conv	3ds-ae-conv_0002-6337051-00_200508-232722
AE-L2	Conv	3ds-ae-b1-conv_0002-6337051-01_200508-232718
VAE	Conv	3ds-vae-b1-conv_0002-6337051-03_200508-232715
2VAE	Conv	3ds-vae-b2-conv_0002-6337051-04_200508-232711
4VAE	Conv	3ds-vae-b4-conv_0002-6337051-05_200508-232713
8VAE	Conv	3ds-vae-b8-conv_0002-6337051-06_200508-232748
16VAE	Conv	3ds-vae-b16-conv_0002-6337051-07_200508-232723
32VAE	Conv	3ds-vae-b32-conv_0002-6337051-08_200508-232717
WAE	Conv	3ds-wae-conv_0002-6337051-02_200508-232718
12 Branch	Branch	3ds-ae-12b1-conv_0005-6337809-01_200509-120419
6 Branch	Branch	3ds-ae-6b2-conv_0034-6373616-01_200518-163704
4 Branch	Branch	3ds-ae-4b3-conv_0034-6373616-00_200518-163709
12 Head	Attention	3ds-ae-12b1-12h1k32v32_0001-6337050-04_200508-232612
6 Head	Attention	3ds-ae-12b1-6h1k32v32_0001-6337050-08_200508-232652
4 Head	Attention	3ds-ae-12b1-4h1k32v32_0001-6337050-07_200508-232614
12 LVAE	Ladder	3ds-vae-lddrev12_0042-6464006-07_200522-152910
12 Ladder-AE	Ladder	3ds-ae-lddrev12_0042-6464006-04_200522-152948
6 LVAE	Ladder	3ds-vae-lddrev6_0042-6464006-06_200522-152917
6 Ladder-AE	Ladder	3ds-ae-lddrev6_0042-6464006-03_200522-152912
4 LVAE	Ladder	3ds-vae-lddrev4_0042-6464006-05_200522-152911
4 Ladder-AE	Ladder	3ds-ae-lddrev4_0042-6464006-02_200522-152915
AE	Baseline	3ds-ae-dislib_0003-6337807-00_200509-120112
AE-L2	Baseline	3ds-ae-b1-dislib_0003-6337807-01_200509-120113
VAE	Baseline	3ds-vae-b1-dislib_0003-6337807-03_200509-120115
2VAE	Baseline	3ds-vae-b2-dislib_0003-6337807-04_200509-120115
4VAE	Baseline	3ds-vae-b4-dislib_0003-6337807-05_200509-120125
8VAE	Baseline	3ds-vae-b8-dislib_0003-6337807-06_200509-120109
16VAE	Baseline	3ds-vae-b16-dislib_0003-6337807-07_200509-120113
32VAE	Baseline	3ds-vae-b32-dislib_0003-6337807-08_200509-120119
WAE	Baseline	3ds-wae-dislib_0003-6337807-02_200509-120136'''

info = util.Table()

# for line in raw.split('\n'):
# 	n,t,r = line.split('\t')
# 	info.new(name=n, type=t, rname=r)
# print(len(info), list(info.select('name')))


for name in os.listdir(root):
	info.new(rname=name)
print(len(info))

dest_root = '/media/fleeb/TRAVELER/completed/'
print(dest_root)

for run in info:
	rpath = os.path.join(root, run.rname)
	vital_names = ['config.yml', 'results.pth.tar', 'eval.pth.tar', 'final.pth.tar']
	
	content_names = set(os.listdir(rpath))
	
	if 'final.pth.tar' in content_names:
		print(run.rname)
		
		ckpts = [(fname, int(fname.split('.')[0].split('_')[-1]))
				 for fname in content_names if 'checkpoint' in fname]
		ckpts = sorted(ckpts, key=lambda x: x[1])
		last_ckpt = ckpts[-1][0]
		
		vital_names.append(last_ckpt)
		
		dpath = os.path.join(dest_root, run.rname)
		if not os.path.isdir(dpath):
			os.mkdir(dpath)
		dest_content = set(os.listdir(dpath))
		for name in vital_names:
			src = os.path.join(rpath, name)
			if name in content_names and name not in dest_content:
				if name == 'eval.pth.tar':
					name = 'results.pth.tar'
				shutil.copy(src, os.path.join(dpath, name))
			# else:
			#     print(f'Skipping {src}')

	else:
		print(f'***MISSING: {run.rname}')
		pass
		
