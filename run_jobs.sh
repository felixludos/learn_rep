


#unbuffer python project.py a/double 3dshapes m/ae --info.extra dbl
#unbuffer python project.py a/double 3dshapes m/ae --info.extra b1-dbl --model.reg_wt 1
#unbuffer python project.py a/double 3dshapes m/wae --info.extra dbl
#unbuffer python project.py a/double 3dshapes m/vae --info.extra b1-dbl --model.reg_wt 1
#unbuffer python project.py a/double 3dshapes m/vae --info.extra b4-dbl --model.reg_wt 4
#unbuffer python project.py a/double 3dshapes m/vae --info.extra b16-dbl --model.reg_wt 16


### 52

#unbuffer python project.py a/d/branch12 a/e/conv 3dshapes m/ae --info.extra 12b1
#unbuffer python project.py a/d/branch12 a/e/double 3dshapes m/ae --info.extra 12b1-dbl


### 53-54

#unbuffer python project.py a/dislib t/3ds-shapes m/ae --info.extra dislib
#unbuffer python project.py a/dislib t/3ds-shapes m/vae --info.extra b1-dislib --model.reg_wt 1
#unbuffer python project.py a/dislib t/3ds-shapes m/vae --info.extra b16-dislib --model.reg_wt 16

#unbuffer python project.py a/conv t/3ds-shapes m/ae --info.extra conv
#unbuffer python project.py a/conv t/3ds-shapes m/vae --info.extra b1-conv --model.reg_wt 1
#unbuffer python project.py a/conv t/3ds-shapes m/vae --info.extra b16-conv --model.reg_wt 16

#unbuffer python project.py a/double t/3ds-shapes m/ae --info.extra dbl
#unbuffer python project.py a/double m/vae t/3ds-shapes --info.extra b1-dbl --model.reg_wt 1
#unbuffer python project.py a/double m/vae t/3ds-shapes --info.extra b16-dbl --model.reg_wt 16

#unbuffer python project.py a/d/branch12 a/e/dislib t/3ds-shapes m/ae --info.extra 12b1-dislib
#unbuffer python project.py a/d/branch12 a/e/conv t/3ds-shapes m/ae --info.extra 12b1-conv
#unbuffer python project.py a/d/branch12 a/e/double t/3ds-shapes m/ae --info.extra 12b1-dbl


### 55-56,58

#unbuffer python project.py t/upd --load "t3ds-shapes-ae-dislib_0053-6286540-00_200429-032431"
#unbuffer python project.py t/upd --load "t3ds-shapes-ae-conv_0053-6286540-03_200429-040022"
#unbuffer python project.py t/upd --load "t3ds-shapes-ae-12b1-conv_0053-6286540-10_200429-041524"

### 57

#unbuffer python project.py a/double m/vae t/3ds-shapes --info.extra b1-dbl --model.reg_wt 1
#unbuffer python project.py a/double m/vae t/3ds-shapes --info.extra b16-dbl --model.reg_wt 16


### 59

#unbuffer python project.py a/dislib t/3ds-shapes m/ae --info.extra dislib-t10k --train_limit 10000
#unbuffer python project.py a/conv t/3ds-shapes m/ae --info.extra conv-t10k --train_limit 10000
#unbuffer python project.py a/d/branch12 a/e/conv t/3ds-shapes m/ae --info.extra 12b1-conv-t10k --train_limit 10000
#
#unbuffer python project.py a/dislib t/3ds-shapes m/ae --info.extra dislib-t25k --train_limit 25000
#unbuffer python project.py a/conv t/3ds-shapes m/ae --info.extra conv-t25k --train_limit 25000
#unbuffer python project.py a/d/branch12 a/e/conv t/3ds-shapes m/ae --info.extra 12b1-conv-t25k --train_limit 25000
#
#unbuffer python project.py a/dislib t/3ds-shapes m/ae --info.extra dislib-t50k --train_limit 50000
#unbuffer python project.py a/conv t/3ds-shapes m/ae --info.extra conv-t50k --train_limit 50000
#unbuffer python project.py a/d/branch12 a/e/conv t/3ds-shapes m/ae --info.extra 12b1-conv-t50k --train_limit 50000
#
#unbuffer python project.py a/dislib t/3ds-shapes m/ae --info.extra dislib-t75k --train_limit 75000
#unbuffer python project.py a/conv t/3ds-shapes m/ae --info.extra conv-t75k --train_limit 75000
#unbuffer python project.py a/d/branch12 a/e/conv t/3ds-shapes m/ae --info.extra 12b1-conv-t75k --train_limit 75000
#
#unbuffer python project.py a/dislib t/3ds-shapes m/ae --info.extra dislib-t100k --train_limit 100000
#unbuffer python project.py a/conv t/3ds-shapes m/ae --info.extra conv-t100k --train_limit 100000
#unbuffer python project.py a/d/branch12 a/e/conv t/3ds-shapes m/ae --info.extra 12b1-conv-t100k --train_limit 100000




