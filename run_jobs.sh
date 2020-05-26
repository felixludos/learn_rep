


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


#################


#unbuffer python project.py a/e/attn12 a/d/branch12 3dshapes m/ae --info.extra 12b1-12h1k64v64 --model.keys_per_head 1 --model.key_val_dim 64 --model.val_dim 32
#unbuffer python project.py a/e/attn12 a/d/branch12 3dshapes m/ae --info.extra 12b1-12h2k64v64 --model.keys_per_head 2 --model.key_val_dim 64 --model.val_dim 32
#unbuffer python project.py a/e/attn12 a/d/branch12 3dshapes m/ae --info.extra 12b1-12h4k64v64 --model.keys_per_head 4 --model.key_val_dim 64 --model.val_dim 32
#
#unbuffer python project.py a/e/attn12 a/d/branch12 3dshapes m/ae --info.extra 12b1-12h1k64v64 --model.keys_per_head 1 --model.key_val_dim 128 --model.val_dim 64
#unbuffer python project.py a/e/attn12 a/d/branch12 3dshapes m/ae --info.extra 12b1-12h1k32v32 --model.keys_per_head 1 --model.key_val_dim 64 --model.val_dim 32
#unbuffer python project.py a/e/attn12 a/d/branch12 3dshapes m/ae --info.extra 12b1-12h1k16v16 --model.keys_per_head 1 --model.key_val_dim 32 --model.val_dim 16
#unbuffer python project.py a/e/attn12 a/d/branch12 3dshapes m/ae --info.extra 12b1-12h1k8v8 --model.keys_per_head 1 --model.key_val_dim 16 --model.val_dim 8
#
#unbuffer python project.py a/e/attn4 a/d/branch12 3dshapes m/ae --info.extra 12b1-4h1k32v32 --model.keys_per_head 1 --model.key_val_dim 64 --model.val_dim 32
#unbuffer python project.py a/e/attn6 a/d/branch12 3dshapes m/ae --info.extra 12b1-6h1k32v32 --model.keys_per_head 1 --model.key_val_dim 64 --model.val_dim 32

unbuffer python project.py a/conv 3dshapes m/ae --info.extra conv
unbuffer python project.py a/conv 3dshapes m/ae --info.extra b1-conv --model.reg_wt 1
unbuffer python project.py a/conv 3dshapes m/wae --info.extra conv
unbuffer python project.py a/conv 3dshapes m/vae --info.extra b1-conv --model.reg_wt 1
unbuffer python project.py a/conv 3dshapes m/vae --info.extra b2-conv --model.reg_wt 2
unbuffer python project.py a/conv 3dshapes m/vae --info.extra b4-conv --model.reg_wt 4
unbuffer python project.py a/conv 3dshapes m/vae --info.extra b8-conv --model.reg_wt 8
unbuffer python project.py a/conv 3dshapes m/vae --info.extra b16-conv --model.reg_wt 16
unbuffer python project.py a/conv 3dshapes m/vae --info.extra b32-conv --model.reg_wt 32

unbuffer python project.py a/dislib 3dshapes m/ae --info.extra dislib
unbuffer python project.py a/dislib 3dshapes m/ae --info.extra b1-dislib --model.reg_wt 1
unbuffer python project.py a/dislib 3dshapes m/wae --info.extra dislib
unbuffer python project.py a/dislib 3dshapes m/vae --info.extra b1-dislib --model.reg_wt 1
unbuffer python project.py a/dislib 3dshapes m/vae --info.extra b2-dislib --model.reg_wt 2
unbuffer python project.py a/dislib 3dshapes m/vae --info.extra b4-dislib --model.reg_wt 4
unbuffer python project.py a/dislib 3dshapes m/vae --info.extra b8-dislib --model.reg_wt 8
unbuffer python project.py a/dislib 3dshapes m/vae --info.extra b16-dislib --model.reg_wt 16
unbuffer python project.py a/dislib 3dshapes m/vae --info.extra b32-dislib --model.reg_wt 32

unbuffer python project.py a/double 3dshapes m/ae --info.extra dbl
unbuffer python project.py a/double 3dshapes m/ae --info.extra b1-dbl --model.reg_wt 1
unbuffer python project.py a/double 3dshapes m/wae --info.extra dbl
unbuffer python project.py a/double 3dshapes m/vae --info.extra b1-dbl --model.reg_wt 1
unbuffer python project.py a/double 3dshapes m/vae --info.extra b2-dbl --model.reg_wt 2
unbuffer python project.py a/double 3dshapes m/vae --info.extra b4-dbl --model.reg_wt 4
unbuffer python project.py a/double 3dshapes m/vae --info.extra b8-dbl --model.reg_wt 8
unbuffer python project.py a/double 3dshapes m/vae --info.extra b16-dbl --model.reg_wt 16
unbuffer python project.py a/double 3dshapes m/vae --info.extra b32-dbl --model.reg_wt 32

###

unbuffer python project.py a/d/branch12 a/e/lib-conv 3dshapes m/ae --info.extra 12b1-dislib
unbuffer python project.py a/d/branch12 a/e/conv 3dshapes m/ae --info.extra 12b1-conv
unbuffer python project.py a/d/branch12 a/e/double 3dshapes m/ae --info.extra 12b1-dbl

###

unbuffer python project.py a/dislib t/3ds-shapes m/ae --info.extra dislib
unbuffer python project.py a/dislib t/3ds-shapes m/vae --info.extra b1-dislib --model.reg_wt 1
unbuffer python project.py a/dislib t/3ds-shapes m/vae --info.extra b16-dislib --model.reg_wt 16

unbuffer python project.py a/conv t/3ds-shapes m/ae --info.extra conv
unbuffer python project.py a/conv t/3ds-shapes m/vae --info.extra b1-conv --model.reg_wt 1
unbuffer python project.py a/conv t/3ds-shapes m/vae --info.extra b16-conv --model.reg_wt 16

unbuffer python project.py a/double t/3ds-shapes m/ae --info.extra dbl
unbuffer python project.py a/double t/3ds-shapes m/vae --info.extra b1-dbl --model.reg_wt 1
unbuffer python project.py a/double t/3ds-shapes m/vae --info.extra b16-dbl --model.reg_wt 16

unbuffer python project.py a/d/branch12 a/e/lib-conv t/3ds-shapes m/ae --info.extra 12b1-dislib
unbuffer python project.py a/d/branch12 a/e/conv t/3ds-shapes m/ae --info.extra 12b1-conv
unbuffer python project.py a/d/branch12 a/e/double t/3ds-shapes m/ae --info.extra 12b1-dbl

###

unbuffer python project.py a/d/branch4 a/e/lib-conv 3dshapes m/ae --info.extra 4b3-dislib
unbuffer python project.py a/d/branch4 a/e/conv 3dshapes m/ae --info.extra 4b3-conv
unbuffer python project.py a/d/branch4 a/e/double 3dshapes m/ae --info.extra 4b3-dbl

unbuffer python project.py a/d/branch6 a/e/lib-conv 3dshapes m/ae --info.extra 6b2-dislib
unbuffer python project.py a/d/branch6 a/e/conv 3dshapes m/ae --info.extra 6b2-conv
unbuffer python project.py a/d/branch6 a/e/double 3dshapes m/ae --info.extra 6b2-dbl

###

unbuffer python project.py t/upd --load t3ds-shapes-ae-12b1-conv_0008-6337836-10_200509-130720
unbuffer python project.py t/upd --load t3ds-shapes-vae-b16-dbl_0008-6337836-08_200509-130723
unbuffer python project.py t/upd --load t3ds-shapes-ae-dbl_0008-6337836-06_200509-130725
unbuffer python project.py t/upd --load t3ds-shapes-ae-conv_0008-6337836-03_200509-130726
unbuffer python project.py t/upd --load t3ds-shapes-vae-b16-conv_0008-6337836-05_200509-130729
unbuffer python project.py t/upd --load t3ds-shapes-vae-b1-dislib_0008-6337836-01_200509-130731
unbuffer python project.py t/upd --load t3ds-shapes-vae-b1-conv_0008-6337836-04_200509-130733
unbuffer python project.py t/upd --load t3ds-shapes-ae-dislib_0008-6337836-00_200509-130737
unbuffer python project.py t/upd --load t3ds-shapes-vae-b16-dislib_0008-6337836-02_200509-130744
unbuffer python project.py t/upd --load t3ds-shapes-ae-12b1-dislib_0008-6337836-09_200509-130759

###

unbuffer python project.py a/conv mpi3d m/ae --dataset.category toy --info.extra conv
unbuffer python project.py a/conv mpi3d m/ae --dataset.category toy --info.extra b1-conv --model.reg_wt 1
unbuffer python project.py a/conv mpi3d m/wae --dataset.category toy --info.extra conv
unbuffer python project.py a/conv mpi3d m/vae --dataset.category toy --info.extra b1-conv --model.reg_wt 1
unbuffer python project.py a/conv mpi3d m/vae --dataset.category toy --info.extra b2-conv --model.reg_wt 2
unbuffer python project.py a/conv mpi3d m/vae --dataset.category toy --info.extra b4-conv --model.reg_wt 4
unbuffer python project.py a/conv mpi3d m/vae --dataset.category toy --info.extra b8-conv --model.reg_wt 8
unbuffer python project.py a/conv mpi3d m/vae --dataset.category toy --info.extra b16-conv --model.reg_wt 16

unbuffer python project.py a/d/branch12 a/e/lib-conv mpi3d m/ae --dataset.category toy --info.extra 12b1-dislib
unbuffer python project.py a/d/branch12 a/e/conv mpi3d m/ae --dataset.category toy --info.extra 12b1-conv
unbuffer python project.py a/d/branch12 a/e/double mpi3d m/ae --dataset.category toy --info.extra 12b1-dbl


unbuffer python project.py a/conv mpi3d m/ae --dataset.category real --info.extra conv
unbuffer python project.py a/conv mpi3d m/ae --dataset.category real --info.extra b1-conv --model.reg_wt 1
unbuffer python project.py a/conv mpi3d m/wae --dataset.category real --info.extra conv
unbuffer python project.py a/conv mpi3d m/vae --dataset.category real --info.extra b1-conv --model.reg_wt 1
unbuffer python project.py a/conv mpi3d m/vae --dataset.category real --info.extra b2-conv --model.reg_wt 2
unbuffer python project.py a/conv mpi3d m/vae --dataset.category real --info.extra b4-conv --model.reg_wt 4
unbuffer python project.py a/conv mpi3d m/vae --dataset.category real --info.extra b8-conv --model.reg_wt 8
unbuffer python project.py a/conv mpi3d m/vae --dataset.category real --info.extra b16-conv --model.reg_wt 16

unbuffer python project.py a/d/branch12 a/e/lib-conv mpi3d m/ae --dataset.category real --info.extra 12b1-dislib
unbuffer python project.py a/d/branch12 a/e/conv mpi3d m/ae --dataset.category real --info.extra 12b1-conv
unbuffer python project.py a/d/branch12 a/e/double mpi3d m/ae --dataset.category real --info.extra 12b1-dbl


unbuffer python project.py a/e/attn12 a/d/branch12 mpi3d m/ae --dataset.category toy --info.extra 12b1-12h1k64v64 --model.keys_per_head 1 --model.key_val_dim 64 --model.val_dim 32
unbuffer python project.py a/e/attn12 a/d/branch12 mpi3d m/ae --dataset.category toy --info.extra 12b1-12h2k64v64 --model.keys_per_head 2 --model.key_val_dim 64 --model.val_dim 32
unbuffer python project.py a/e/attn12 a/d/branch12 mpi3d m/ae --dataset.category toy --info.extra 12b1-12h4k64v64 --model.keys_per_head 4 --model.key_val_dim 64 --model.val_dim 32

unbuffer python project.py a/e/attn12 a/d/branch12 mpi3d m/ae --dataset.category toy --info.extra 12b1-12h1k64v64 --model.keys_per_head 1 --model.key_val_dim 128 --model.val_dim 64
unbuffer python project.py a/e/attn12 a/d/branch12 mpi3d m/ae --dataset.category toy --info.extra 12b1-12h1k32v32 --model.keys_per_head 1 --model.key_val_dim 64 --model.val_dim 32
unbuffer python project.py a/e/attn12 a/d/branch12 mpi3d m/ae --dataset.category toy --info.extra 12b1-12h1k16v16 --model.keys_per_head 1 --model.key_val_dim 32 --model.val_dim 16
unbuffer python project.py a/e/attn12 a/d/branch12 mpi3d m/ae --dataset.category toy --info.extra 12b1-12h1k8v8 --model.keys_per_head 1 --model.key_val_dim 16 --model.val_dim 8

unbuffer python project.py a/e/attn4 a/d/branch12 mpi3d m/ae --dataset.category toy --info.extra 12b1-4h1k32v32 --model.keys_per_head 1 --model.key_val_dim 64 --model.val_dim 32
unbuffer python project.py a/e/attn6 a/d/branch12 mpi3d m/ae --dataset.category toy --info.extra 12b1-6h1k32v32 --model.keys_per_head 1 --model.key_val_dim 64 --model.val_dim 32

###

unbuffer python project.py a/e/attn12 a/d/branch12 mpi3d m/ae --dataset.category real --info.extra 12b1-12h1k64v64 --model.keys_per_head 1 --model.key_val_dim 64 --model.val_dim 32
unbuffer python project.py a/e/attn12 a/d/branch12 mpi3d m/ae --dataset.category real --info.extra 12b1-12h2k64v64 --model.keys_per_head 2 --model.key_val_dim 64 --model.val_dim 32
unbuffer python project.py a/e/attn12 a/d/branch12 mpi3d m/ae --dataset.category real --info.extra 12b1-12h4k64v64 --model.keys_per_head 4 --model.key_val_dim 64 --model.val_dim 32

unbuffer python project.py a/e/attn12 a/d/branch12 mpi3d m/ae --dataset.category real --info.extra 12b1-12h1k64v64 --model.keys_per_head 1 --model.key_val_dim 128 --model.val_dim 64
unbuffer python project.py a/e/attn12 a/d/branch12 mpi3d m/ae --dataset.category real --info.extra 12b1-12h1k32v32 --model.keys_per_head 1 --model.key_val_dim 64 --model.val_dim 32
unbuffer python project.py a/e/attn12 a/d/branch12 mpi3d m/ae --dataset.category real --info.extra 12b1-12h1k16v16 --model.keys_per_head 1 --model.key_val_dim 32 --model.val_dim 16
unbuffer python project.py a/e/attn12 a/d/branch12 mpi3d m/ae --dataset.category real --info.extra 12b1-12h1k8v8 --model.keys_per_head 1 --model.key_val_dim 16 --model.val_dim 8

unbuffer python project.py a/e/attn4 a/d/branch12 mpi3d m/ae --dataset.category real --info.extra 12b1-4h1k32v32 --model.keys_per_head 1 --model.key_val_dim 64 --model.val_dim 32
unbuffer python project.py a/e/attn6 a/d/branch12 mpi3d m/ae --dataset.category real --info.extra 12b1-6h1k32v32 --model.keys_per_head 1 --model.key_val_dim 64 --model.val_dim 32

#unbuffer python project.py a/double t/mpi-shapes m/ae --dataset.category toy --info.extra toy-dbl
#unbuffer python project.py a/double t/mpi-shapes m/vae --dataset.category toy --info.extra toy-b1-dbl --model.reg_wt 1
#unbuffer python project.py a/double t/mpi-shapes m/vae --dataset.category toy --info.extra toy-b2-dbl --model.reg_wt 2
#
#unbuffer python project.py a/d/branch12 a/e/lib-conv t/mpi-shapes m/ae --info.extra toy-12b1-dislib
#unbuffer python project.py a/d/branch12 a/e/double t/mpi-shapes m/ae --info.extra toy-12b1-dbl

############################################# MPI transfer

unbuffer python project.py a/d/branch12 a/e/conv t/mpi-sizes m/ae --info.extra toy-12b1-conv
unbuffer python project.py a/conv t/mpi-sizes m/vae --dataset.category toy --info.extra toy-b2-conv --model.reg_wt 2
unbuffer python project.py a/dislib t/mpi-sizes m/vae --dataset.category toy --info.extra toy-b2-dislib --model.reg_wt 2

unbuffer python project.py a/dislib t/mpi-sizes m/ae --dataset.category toy --info.extra toy-dislib
unbuffer python project.py a/dislib t/mpi-sizes m/vae --dataset.category toy --info.extra toy-b1-dislib --model.reg_wt 1

unbuffer python project.py a/conv t/mpi-sizes m/ae --dataset.category toy --info.extra toy-conv
unbuffer python project.py a/conv t/mpi-sizes m/vae --dataset.category toy --info.extra toy-b1-conv --model.reg_wt 1

unbuffer python project.py a/d/branch12 a/e/conv t/mpi-sizes m/ae --info.extra toy-12b1-conv-s2 --seed 2
unbuffer python project.py a/conv t/mpi-sizes m/vae --dataset.category toy --info.extra toy-b2-conv-s2 --model.reg_wt 2 --seed 2
unbuffer python project.py a/dislib t/mpi-sizes m/vae --dataset.category toy --info.extra toy-b2-dislib-s2 --model.reg_wt 2 --seed 2

###

unbuffer python project.py a/d/branch12 a/e/conv t/mpi-shapes m/ae --info.extra toy-12b1-conv
unbuffer python project.py a/conv t/mpi-shapes m/vae --dataset.category toy --info.extra toy-b2-conv --model.reg_wt 2
unbuffer python project.py a/dislib t/mpi-shapes m/vae --dataset.category toy --info.extra toy-b2-dislib --model.reg_wt 2

unbuffer python project.py a/dislib t/mpi-shapes m/ae --dataset.category toy --info.extra toy-dislib
unbuffer python project.py a/dislib t/mpi-shapes m/vae --dataset.category toy --info.extra toy-b1-dislib --model.reg_wt 1

unbuffer python project.py a/conv t/mpi-shapes m/ae --dataset.category toy --info.extra toy-conv
unbuffer python project.py a/conv t/mpi-shapes m/vae --dataset.category toy --info.extra toy-b1-conv --model.reg_wt 1

unbuffer python project.py a/d/branch12 a/e/conv t/mpi-shapes m/ae --info.extra toy-12b1-conv-s2 --seed 2
unbuffer python project.py a/conv t/mpi-shapes m/vae --dataset.category toy --info.extra toy-b2-conv-s2 --model.reg_wt 2 --seed 2
unbuffer python project.py a/dislib t/mpi-shapes m/vae --dataset.category toy --info.extra toy-b2-dislib-s2 --model.reg_wt 2 --seed 2

###

unbuffer python project.py a/d/branch12 a/e/conv t/mpi-sizes m/ae --dataset.category real --info.extra real-12b1-conv
unbuffer python project.py a/conv t/mpi-sizes m/vae --dataset.category real --info.extra real-b2-conv --model.reg_wt 2
unbuffer python project.py a/dislib t/mpi-sizes m/vae --dataset.category real --info.extra real-b2-dislib --model.reg_wt 2

unbuffer python project.py a/dislib t/mpi-sizes m/ae --dataset.category real --info.extra real-dislib
unbuffer python project.py a/dislib t/mpi-sizes m/vae --dataset.category real --info.extra real-b1-dislib --model.reg_wt 1

unbuffer python project.py a/conv t/mpi-sizes m/ae --dataset.category real --info.extra real-conv
unbuffer python project.py a/conv t/mpi-sizes m/vae --dataset.category real --info.extra real-b1-conv --model.reg_wt 1

unbuffer python project.py a/d/branch12 a/e/conv t/mpi-sizes m/ae --info.extra real-12b1-conv-s2 --seed 2
unbuffer python project.py a/conv t/mpi-sizes m/vae --dataset.category real --info.extra real-b2-conv-s2 --model.reg_wt 2 --seed 2
unbuffer python project.py a/dislib t/mpi-sizes m/vae --dataset.category real --info.extra real-b2-dislib-s2 --model.reg_wt 2 --seed 2

###

unbuffer python project.py a/d/branch12 a/e/conv t/mpi-shapes m/ae --dataset.category real --info.extra real-12b1-conv
unbuffer python project.py a/conv t/mpi-shapes m/vae --dataset.category real --info.extra real-b2-conv --model.reg_wt 2
unbuffer python project.py a/dislib t/mpi-shapes m/vae --dataset.category real --info.extra real-b2-dislib --model.reg_wt 2

unbuffer python project.py a/dislib t/mpi-shapes m/ae --dataset.category real --info.extra real-dislib
unbuffer python project.py a/dislib t/mpi-shapes m/vae --dataset.category real --info.extra real-b1-dislib --model.reg_wt 1

unbuffer python project.py a/conv t/mpi-shapes m/ae --dataset.category real --info.extra real-conv
unbuffer python project.py a/conv t/mpi-shapes m/vae --dataset.category real --info.extra real-b1-conv --model.reg_wt 1

unbuffer python project.py a/d/branch12 a/e/conv t/mpi-shapes m/ae --info.extra real-12b1-conv-s2 --seed 2
unbuffer python project.py a/conv t/mpi-shapes m/vae --dataset.category real --info.extra real-b2-conv-s2 --model.reg_wt 2 --seed 2
unbuffer python project.py a/dislib t/mpi-shapes m/vae --dataset.category real --info.extra real-b2-dislib-s2 --model.reg_wt 2 --seed 2

###

unbuffer python project.py long a/d/lbranch12 a/e/conv mpi3d m/ae --info.extra l-toy-12b1-conv
unbuffer python project.py long a/lconv mpi3d m/vae --dataset.category toy --info.extra l-toy-b2-conv --model.reg_wt 2

unbuffer python project.py long a/lconv mpi3d m/ae --dataset.category toy --info.extra l-toy-conv
unbuffer python project.py long a/lconv mpi3d m/vae --dataset.category toy --info.extra l-toy-b1-conv --model.reg_wt 1
unbuffer python project.py long a/lconv mpi3d m/vae --dataset.category toy --info.extra l-toy-b4-conv --model.reg_wt 4

unbuffer python project.py long a/d/lbranch12 a/e/lconv mpi3d m/ae --info.extra l-toy-12b1-conv-s2 --seed 2
unbuffer python project.py long a/lconv mpi3d m/vae --dataset.category toy --info.extra l-toy-b2-conv-s2 --model.reg_wt 2 --seed 2

###

unbuffer python project.py a/d/dbranch12 a/e/dconv celeba m/ae --info.extra 12b1-conv
unbuffer python project.py a/deep celeba m/vae --info.extra b2-conv --model.reg_wt 2

unbuffer python project.py a/deep celeba m/ae --info.extra conv
unbuffer python project.py a/deep celeba m/wae --info.extra conv
unbuffer python project.py a/deep celeba m/vae --info.extra b1-conv --model.reg_wt 1
unbuffer python project.py a/deep celeba m/vae --info.extra b4-conv --model.reg_wt 4
unbuffer python project.py a/deep celeba m/vae --info.extra b16-conv --model.reg_wt 16

unbuffer python project.py a/d/dbranch12 a/e/dconv celeba m/ae --info.extra 12b1-conv-s2 --seed 2
unbuffer python project.py a/deep celeba m/vae --info.extra b2-conv-s2 --model.reg_wt 2 --seed 2

###

unbuffer python project.py a/d/dbranch12 a/e/dconv celeba m/ae --info.extra 12b1-conv
unbuffer python project.py a/deep celeba m/vae --info.extra b2-conv --model.reg_wt 2

unbuffer python project.py a/deep celeba m/ae --info.extra conv
unbuffer python project.py a/deep celeba m/wae --info.extra conv
unbuffer python project.py a/deep celeba m/vae --info.extra b1-conv --model.reg_wt 1
unbuffer python project.py a/deep celeba m/vae --info.extra b4-conv --model.reg_wt 4
unbuffer python project.py a/deep celeba m/vae --info.extra b16-conv --model.reg_wt 16

unbuffer python project.py a/d/dbranch12 a/e/dconv celeba m/ae --info.extra 12b1-conv-s2 --seed 2
unbuffer python project.py a/deep celeba m/vae --info.extra b2-conv-s2 --model.reg_wt 2 --seed 2

###

unbuffer python project.py a/d/branch12 a/e/conv 3dshapes m/ae --info.extra 12b1-conv --seed 10


unbuffer python project.py a/d/branch12 a/e/conv 3dshapes m/ae --info.extra 12b1-conv --seed 10
unbuffer python project.py a/conv 3dshapes m/ae --info.extra conv --seed 10
unbuffer python project.py a/conv 3dshapes m/wae --info.extra conv --seed 10
unbuffer python project.py a/conv 3dshapes m/vae --info.extra b1-conv --model.reg_wt 1 --seed 10
unbuffer python project.py a/conv 3dshapes m/vae --info.extra b16-conv --model.reg_wt 16 --seed 10


###

unbuffer python project.py a/e/attn12 a/d/branch12 3dshapes m/ae --info.extra 12b1-12h1k32v32 --seed 10
unbuffer python project.py a/e/attn4 a/d/branch12 3dshapes m/ae --info.extra 12b1-4h1k32v32 --seed 10
unbuffer python project.py a/e/attn6 a/d/branch12 3dshapes m/ae --info.extra 12b1-6h1k32v32 --seed 10

unbuffer python project.py a/e/attn12 a/d/deconv 3dshapes m/vae --info.extra b16-conv-12h1k32v32 --model.reg_wt 16 --seed 10
unbuffer python project.py a/e/attn6 a/d/deconv 3dshapes m/vae --info.extra b16-conv-6h1k32v32 --model.reg_wt 16 --seed 10
unbuffer python project.py a/e/attn12 a/d/deconv 3dshapes m/vae --info.extra b1-conv-12h1k32v32 --model.reg_wt 1 --seed 10
unbuffer python project.py a/e/attn6 a/d/deconv 3dshapes m/vae --info.extra b1-conv-6h1k32v32 --model.reg_wt 1 --seed 10

unbuffer python project.py a/e/attn12 a/d/deconv 3dshapes m/ae --info.extra conv-12h1k32v32 --seed 10
unbuffer python project.py a/e/attn4 a/d/deconv 3dshapes m/ae --info.extra conv-4h1k32v32 --seed 10
unbuffer python project.py a/e/attn6 a/d/deconv 3dshapes m/ae --info.extra conv-6h1k32v32 --seed 10

###


unbuffer python project.py a/d/branch4 a/e/conv t/mpi-shapes m/ae --dataset.category real --info.extra real-4b3-conv
unbuffer python project.py a/d/branch6 a/e/conv t/mpi-shapes m/ae --dataset.category real --info.extra real-6b2-conv
#unbuffer python project.py a/d/branch12 a/e/conv t/mpi-shapes m/ae --dataset.category real --info.extra real-12b1-conv

#unbuffer python project.py a/conv t/mpi-shapes m/vae --dataset.category real --info.extra real-b2-conv --model.reg_wt 2
#unbuffer python project.py a/dislib t/mpi-shapes m/vae --dataset.category real --info.extra real-b2-dislib --model.reg_wt 2

unbuffer python project.py a/dislib t/mpi-shapes m/wae --dataset.category real --info.extra real-dislib
#unbuffer python project.py a/dislib t/mpi-shapes m/ae --dataset.category real --info.extra real-dislib
#unbuffer python project.py a/dislib t/mpi-shapes m/vae --dataset.category real --info.extra real-b1-dislib --model.reg_wt 1

unbuffer python project.py a/conv t/mpi-shapes m/wae --dataset.category real --info.extra real-conv
#unbuffer python project.py a/conv t/mpi-shapes m/ae --dataset.category real --info.extra real-conv
#unbuffer python project.py a/conv t/mpi-shapes m/vae --dataset.category real --info.extra real-b1-conv --model.reg_wt 1


###

unbuffer python project.py a/d/branch4 a/e/conv t/mpi-shapes m/ae --dataset.category real --info.extra real-4b3-conv
unbuffer python project.py a/d/branch6 a/e/conv t/mpi-shapes m/ae --dataset.category real --info.extra real-6b2-conv
unbuffer python project.py a/dislib t/mpi-shapes m/wae --dataset.category real --info.extra real-dislib
unbuffer python project.py a/conv t/mpi-shapes m/wae --dataset.category real --info.extra real-conv

unbuffer python project.py a/d/branch4 a/e/conv t/mpi-shapes m/ae --dataset.category toy --info.extra toy-4b3-conv
unbuffer python project.py a/d/branch6 a/e/conv t/mpi-shapes m/ae --dataset.category toy --info.extra toy-6b2-conv
unbuffer python project.py a/dislib t/mpi-shapes m/wae --dataset.category toy --info.extra toy-dislib
unbuffer python project.py a/conv t/mpi-shapes m/wae --dataset.category toy --info.extra toy-conv

unbuffer python project.py a/d/branch4 a/e/conv t/3ds-shapes m/ae --info.extra 4b3-conv
unbuffer python project.py a/d/branch6 a/e/conv t/3ds-shapes m/ae --info.extra 6b2-conv
unbuffer python project.py a/dislib t/3ds-shapes m/wae --info.extra dislib
unbuffer python project.py a/conv t/3ds-shapes m/wae --info.extra conv

#unbuffer python project.py a/d/branch12 a/e/conv t/3ds-shapes m/ae --info.extra 12b1-conv-t10k --train_limit 10000

###

unbuffer python project.py resbranch a/d/dbranch12 a/e/dconv celeba m/ae --info.extra r12b1-conv

unbuffer python project.py resbranch a/d/branch12 a/e/dconv 3dshapes m/ae --info.extra r12b1-conv
unbuffer python project.py resbranch a/d/branch6 a/e/dconv 3dshapes m/ae --info.extra r6b2-conv
unbuffer python project.py resbranch a/d/branch4 a/e/dconv 3dshapes m/ae --info.extra r4b3-conv

unbuffer python project.py resbranch a/d/branch12 a/e/dconv mpi3d m/ae --dataset.category toy --info.extra toy-r12b1-conv
unbuffer python project.py resbranch a/d/branch6 a/e/dconv mpi3d m/ae --dataset.category toy --info.extra toy-r6b2-conv
unbuffer python project.py resbranch a/d/branch4 a/e/dconv mpi3d m/ae --dataset.category toy --info.extra toy-r4b3-conv

unbuffer python project.py resbranch a/d/branch12 a/e/dconv mpi3d m/ae --dataset.category real --info.extra real-r12b1-conv
unbuffer python project.py resbranch a/d/branch6 a/e/dconv mpi3d m/ae --dataset.category real --info.extra real-r6b2-conv
unbuffer python project.py resbranch a/d/branch4 a/e/dconv mpi3d m/ae --dataset.category real --info.extra real-r4b3-conv


###


unbuffer python project.py a/d/branch12 a/e/attn12 t/3ds-shapes m/ae --info.extra 12b1-12h-conv
unbuffer python project.py a/d/branch12 a/e/attn6 t/3ds-shapes m/ae --info.extra 12b1-6h-conv
unbuffer python project.py a/d/branch12 a/e/attn4 t/3ds-shapes m/ae --info.extra 12b1-4h-conv

unbuffer python project.py a/d/branch12 a/e/attn12 t/mpi-shapes m/ae --dataset.category toy --info.extra toy-12b1-12h-conv
unbuffer python project.py a/d/branch12 a/e/attn6 t/mpi-shapes m/ae --dataset.category toy --info.extra toy-12b1-6h-conv
unbuffer python project.py a/d/branch12 a/e/attn4 t/mpi-shapes m/ae --dataset.category toy --info.extra toy-12b1-4h-conv

unbuffer python project.py a/d/branch12 a/e/attn12 t/mpi-shapes m/ae --dataset.category real --info.extra real-12b1-12h-conv
unbuffer python project.py a/d/branch12 a/e/attn6 t/mpi-shapes m/ae --dataset.category real --info.extra real-12b1-6h-conv
unbuffer python project.py a/d/branch12 a/e/attn4 t/mpi-shapes m/ae --dataset.category real --info.extra real-12b1-4h-conv

###


unbuffer python project.py a/ladder12 3dshapes m/ae --info.extra r-lddrev12 --model.residual_core

unbuffer python project.py a/ladder4 3dshapes m/ae --info.extra lddrev4
unbuffer python project.py a/ladder6 3dshapes m/ae --info.extra lddrev6
unbuffer python project.py a/ladder12 3dshapes m/ae --info.extra lddrev12

unbuffer python project.py a/ladder4 3dshapes m/vae --info.extra lddrev4
unbuffer python project.py a/ladder6 3dshapes m/vae --info.extra lddrev6
unbuffer python project.py a/ladder12 3dshapes m/vae --info.extra lddrev12

unbuffer python project.py a/ladder4 mpi3d m/ae --dataset.category toy --info.extra lddrev4
unbuffer python project.py a/ladder6 mpi3d m/ae --dataset.category toy --info.extra lddrev6
unbuffer python project.py a/ladder12 mpi3d m/ae --dataset.category toy --info.extra lddrev12

unbuffer python project.py a/ladder4 mpi3d m/vae --dataset.category toy --info.extra lddrev4
unbuffer python project.py a/ladder6 mpi3d m/vae --dataset.category toy --info.extra lddrev6
unbuffer python project.py a/ladder12 mpi3d m/vae --dataset.category toy --info.extra lddrev12

unbuffer python project.py a/ladder4 mpi3d m/vae --dataset.category real --info.extra lddrev4
unbuffer python project.py a/ladder6 mpi3d m/vae --dataset.category real --info.extra lddrev6
unbuffer python project.py a/ladder12 mpi3d m/vae --dataset.category real --info.extra lddrev12

unbuffer python project.py a/ladder16 celeba m/vae --info.extra lddrev12

###

#unbuffer python project.py t/upd --load t3ds-shapes-ae-12b1-12h-conv_0039-6425005-00_200521-150816
#unbuffer python project.py t/upd --load t3ds-shapes-ae-12b1-6h-conv_0039-6425005-01_200521-150812
#unbuffer python project.py t/upd --load t3ds-shapes-ae-12b1-4h-conv_0039-6425005-02_200521-150810
#
#unbuffer python project.py t/upd --load tmpi-shapes-ae-toy-12b1-12h-conv_0039-6425005-03_200521-150814
#unbuffer python project.py t/upd --load tmpi-shapes-ae-toy-12b1-4h-conv_0039-6425005-05_200521-165857
#
#unbuffer python project.py t/upd --load tmpi-shapes-ae-real-12b1-12h-conv_0039-6425005-06_200521-165855
#unbuffer python project.py t/upd --load tmpi-shapes-ae-real-12b1-6h-conv_0039-6425005-07_200521-181623
#unbuffer python project.py t/upd --load tmpi-shapes-ae-real-12b1-4h-conv_0039-6425005-08_200521-181701

###

#unbuffer python project.py t/test --load t3ds-shapes-ae-12b1-conv_0010-6339705-00_200509-233443
#unbuffer python project.py t/test --load t3ds-shapes-ae-conv_0010-6339705-03_200509-233342
#unbuffer python project.py t/test --load t3ds-shapes-vae-b1-conv_0010-6339705-06_200509-233330
#unbuffer python project.py t/test --load t3ds-shapes-vae-b16-conv_0010-6339705-04_200509-233331
#unbuffer python project.py t/test --load t3ds-shapes-ae-6b2-conv_0040-6457811-09_200521-192043
#unbuffer python project.py t/test --load t3ds-shapes-ae-4b3-conv_0040-6457811-08_200521-190411

#unbuffer python project.py t/test --load t3ds-shapes-ae-dislib_0010-6339705-07_200509-233331
#unbuffer python project.py t/test --load t3ds-shapes-vae-b1-dislib_0010-6339705-05_200509-233352
#unbuffer python project.py t/test --load t3ds-shapes-vae-b16-dislib_0010-6339705-08_200509-233335
#unbuffer python project.py t/test --load t3ds-shapes-wae-dislib_0040-6457811-10_200521-192058
#unbuffer python project.py t/test --load t3ds-shapes-wae-conv_0040-6457811-11_200521-192112


#unbuffer python project.py t/test --load tmpi-shapes-ae-toy-conv_0029-6357652-05_200517-020945
#unbuffer python project.py t/test --load tmpi-shapes-vae-toy-b1-conv_0029-6357652-06_200517-021252
#unbuffer python project.py t/test --load tmpi-shapes-vae-toy-b2-conv_0029-6357652-01_200517-015851
#unbuffer python project.py t/test --load tmpi-shapes-ae-toy-12b1-conv_0029-6357652-00_200517-021120
#unbuffer python project.py t/test --load tmpi-shapes-ae-toy-6b2-conv_0040-6457811-06_200521-185626
#unbuffer python project.py t/test --load tmpi-shapes-ae-toy-4b3-conv_0040-6457811-07_200521-190415

#unbuffer python project.py t/test --load tmpi-shapes-ae-toy-dislib_0029-6357652-03_200517-020655
#unbuffer python project.py t/test --load tmpi-shapes-vae-toy-b1-dislib_0029-6357652-04_200517-020750
#unbuffer python project.py t/test --load tmpi-shapes-vae-toy-b2-dislib_0029-6357652-02_200517-020148
#unbuffer python project.py t/test --load tmpi-shapes-wae-toy-dislib_0040-6457811-05_200521-183531
#unbuffer python project.py t/test --load tmpi-shapes-wae-toy-conv_0040-6457811-04_200521-182820


#unbuffer python project.py t/test --load tmpi-shapes-ae-real-conv_0028-6357651-05_200517-013845
#unbuffer python project.py t/test --load tmpi-shapes-vae-real-b1-conv_0028-6357651-06_200517-014147
#unbuffer python project.py t/test --load tmpi-shapes-vae-real-b2-conv_0028-6357651-01_200517-013344
#unbuffer python project.py t/test --load tmpi-shapes-ae-real-12b1-conv_0028-6357651-00_200517-013342
#unbuffer python project.py t/test --load tmpi-shapes-ae-real-6b2-conv_0040-6457811-01_200521-181822
#unbuffer python project.py t/test --load tmpi-shapes-ae-real-4b3-conv_0040-6457811-00_200521-181732

#unbuffer python project.py t/test --load tmpi-shapes-ae-real-dislib_0028-6357651-02_200517-013348
#unbuffer python project.py t/test --load tmpi-shapes-vae-real-b1-dislib_0028-6357651-03_200517-013344
#unbuffer python project.py t/test --load tmpi-shapes-vae-real-b2-dislib_0028-6357651-04_200517-013349
#unbuffer python project.py t/test --load tmpi-shapes-wae-real-dislib_0040-6457811-03_200521-182624
#unbuffer python project.py t/test --load tmpi-shapes-wae-real-conv_0040-6457811-02_200521-181826

###


unbuffer python project.py a/ladder4 t/3ds-shapes m/vae --info.extra lddrev4
unbuffer python project.py a/ladder6 t/3ds-shapes m/vae --info.extra lddrev6
unbuffer python project.py a/ladder12 t/3ds-shapes m/vae --info.extra lddrev12

unbuffer python project.py a/ladder4 t/mpi-shapes m/vae --dataset.category toy --info.extra toy-lddrev4
unbuffer python project.py a/ladder6 t/mpi-shapes m/vae --dataset.category toy --info.extra toy-lddrev6
unbuffer python project.py a/ladder12 t/mpi-shapes m/vae --dataset.category toy --info.extra toy-lddrev12

unbuffer python project.py a/ladder4 t/mpi-shapes m/vae --dataset.category real --info.extra real-lddrev4
unbuffer python project.py a/ladder6 t/mpi-shapes m/vae --dataset.category real --info.extra real-lddrev6
unbuffer python project.py a/ladder12 t/mpi-shapes m/vae --dataset.category real --info.extra real-lddrev12


unbuffer python project.py a/ladder4 t/3ds-shapes m/ae --info.extra lddrev4
unbuffer python project.py a/ladder6 t/3ds-shapes m/ae --info.extra lddrev6
unbuffer python project.py a/ladder12 t/3ds-shapes m/ae --info.extra lddrev12

unbuffer python project.py a/ladder4 t/mpi-shapes m/ae --dataset.category toy --info.extra toy-lddrev4
unbuffer python project.py a/ladder6 t/mpi-shapes m/ae --dataset.category toy --info.extra toy-lddrev6
unbuffer python project.py a/ladder12 t/mpi-shapes m/ae --dataset.category toy --info.extra toy-lddrev12

unbuffer python project.py a/ladder4 t/mpi-shapes m/ae --dataset.category real --info.extra real-lddrev4
unbuffer python project.py a/ladder6 t/mpi-shapes m/ae --dataset.category real --info.extra real-lddrev6
unbuffer python project.py a/ladder12 t/mpi-shapes m/ae --dataset.category real --info.extra real-lddrev12

###

unbuffer python project.py --resume mpi3d-vae-lddrev6_0042-6464006-12_200522-152917

###

unbuffer python project.py a/d/branch4 a/e/conv mpi3d m/ae --dataset.category toy --info.extra toy-4b3
unbuffer python project.py a/d/branch6 a/e/conv mpi3d m/ae --dataset.category toy --info.extra toy-6b2

unbuffer python project.py a/d/branch4 a/e/conv mpi3d m/ae --dataset.category real --info.extra real-4b3
unbuffer python project.py a/d/branch6 a/e/conv mpi3d m/ae --dataset.category real --info.extra real-6b2

###

unbuffer python project.py t/upd --load t3ds-shapes-vae-lddrev4_0050-6465005-00_200523-155658
unbuffer python project.py t/upd --load t3ds-shapes-vae-lddrev6_0050-6465005-01_200523-155659
unbuffer python project.py t/upd --load t3ds-shapes-vae-lddrev12_0050-6465005-02_200523-155700
unbuffer python project.py t/upd --load tmpi-shapes-vae-real-lddrev12_0050-6465005-08_200523-155703
unbuffer python project.py t/upd --load tmpi-shapes-vae-toy-lddrev6_0050-6465005-04_200523-155705
unbuffer python project.py t/upd --load tmpi-shapes-vae-toy-lddrev12_0050-6465005-05_200523-155707
unbuffer python project.py t/upd --load tmpi-shapes-vae-real-lddrev4_0050-6465005-06_200523-155708
unbuffer python project.py t/upd --load tmpi-shapes-vae-real-lddrev6_0050-6465005-07_200523-155709
unbuffer python project.py t/upd --load tmpi-shapes-vae-toy-lddrev4_0050-6465005-03_200523-155710

unbuffer python project.py t/upd --load t3ds-shapes-ae-lddrev4_0051-6465006-00_200523-155726
unbuffer python project.py t/upd --load t3ds-shapes-ae-lddrev12_0051-6465006-02_200523-155728
unbuffer python project.py t/upd --load t3ds-shapes-ae-lddrev6_0051-6465006-01_200523-155730
unbuffer python project.py t/upd --load tmpi-shapes-ae-toy-lddrev4_0051-6465006-03_200523-155735
unbuffer python project.py t/upd --load tmpi-shapes-ae-real-lddrev12_0051-6465006-08_200523-155735
unbuffer python project.py t/upd --load tmpi-shapes-ae-real-lddrev4_0051-6465006-06_200523-155738
unbuffer python project.py t/upd --load tmpi-shapes-ae-real-lddrev6_0051-6465006-07_200523-155739
unbuffer python project.py t/upd --load tmpi-shapes-ae-toy-lddrev12_0051-6465006-05_200523-155740
unbuffer python project.py t/upd --load tmpi-shapes-ae-toy-lddrev6_0051-6465006-04_200523-155742

###

unbuffer python project.py final --resume celeba-vae-b4-conv_0031-6362398-05_200517-170611
unbuffer python project.py final --resume celeba-ae-12b1-conv_0031-6362398-00_200517-170612
unbuffer python project.py final --resume celeba-vae-b1-conv_0031-6362398-04_200517-170613
unbuffer python project.py final --resume celeba-vae-b2-conv-s2_0031-6362398-08_200517-170614
unbuffer python project.py final --resume celeba-ae-12b1-conv-s2_0031-6362398-07_200517-170615
unbuffer python project.py final --resume celeba-vae-b2-conv_0031-6362398-01_200517-170616
unbuffer python project.py final --resume celeba-ae-conv_0031-6362398-02_200517-170621
unbuffer python project.py final --resume celeba-vae-b16-conv_0031-6362398-06_200517-170622

###

unbuffer python project.py t/test --load t3ds-shapes-vae-lddrev12_0063-6470278-02_200525-212130
unbuffer python project.py t/test --load t3ds-shapes-ae-lddrev12_0063-6470278-10_200525-212135
unbuffer python project.py t/test --load t3ds-shapes-vae-lddrev6_0063-6470278-01_200525-212141
unbuffer python project.py t/test --load t3ds-shapes-ae-lddrev6_0063-6470278-11_200525-212130
unbuffer python project.py t/test --load t3ds-shapes-vae-lddrev4_0063-6470278-00_200525-212135
unbuffer python project.py t/test --load t3ds-shapes-ae-lddrev4_0063-6470278-09_200525-212134

unbuffer python project.py t/test --load tmpi-shapes-vae-toy-lddrev12_0063-6470278-05_200525-212135
unbuffer python project.py t/test --load tmpi-shapes-ae-toy-lddrev12_0063-6470278-16_200525-212133
unbuffer python project.py t/test --load tmpi-shapes-vae-toy-lddrev6_0063-6470278-04_200525-212131
unbuffer python project.py t/test --load tmpi-shapes-ae-toy-lddrev6_0063-6470278-17_200525-212134
unbuffer python project.py t/test --load tmpi-shapes-vae-toy-lddrev4_0063-6470278-08_200525-212143
unbuffer python project.py t/test --load tmpi-shapes-ae-toy-lddrev4_0063-6470278-12_200525-212133

unbuffer python project.py t/test --load tmpi-shapes-vae-real-lddrev12_0063-6470278-03_200525-212143
unbuffer python project.py t/test --load tmpi-shapes-ae-real-lddrev12_0063-6470278-13_200525-212137
unbuffer python project.py t/test --load tmpi-shapes-vae-real-lddrev6_0063-6470278-07_200525-212158
unbuffer python project.py t/test --load tmpi-shapes-ae-real-lddrev6_0063-6470278-15_200525-212149
unbuffer python project.py t/test --load tmpi-shapes-vae-real-lddrev4_0063-6470278-06_200525-212137
unbuffer python project.py t/test --load tmpi-shapes-ae-real-lddrev4_0063-6470278-14_200525-212145


#####################

#unbuffer python project.py a/d/branch12 a/e/dislib t/3ds-shapes m/ae --info.extra 12b1-dislib
#unbuffer python project.py a/e/attn12 a/d/branch12 m/ae 3dshapes --info.extra 12b1-12h4k64v64-seed9 --model.keys_per_head 4 --model.key_val_dim 128 --model.val_dim 64 --seed 9
#unbuffer python project.py a/d/branch12 a/e/conv m/ae mpi3d --dataset.category real --info.extra 12b1-seed5 --seed 9





