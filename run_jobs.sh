


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


#####################

#unbuffer python project.py a/d/branch12 a/e/dislib t/3ds-shapes m/ae --info.extra 12b1-dislib
#unbuffer python project.py a/e/attn12 a/d/branch12 m/ae 3dshapes --info.extra 12b1-12h4k64v64-seed9 --model.keys_per_head 4 --model.key_val_dim 128 --model.val_dim 64 --seed 9
#unbuffer python project.py a/d/branch12 a/e/conv m/ae mpi3d --dataset.category real --info.extra 12b1-seed5 --seed 9





