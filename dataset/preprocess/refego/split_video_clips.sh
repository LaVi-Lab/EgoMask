#!/bin/bash

# Reference: https://github.com/shuheikurita/RefEgo/blob/main/dataset/split_video_clips.sh

# Extracting images_extracted_fps2/ from images_half_fps2/ 

if [ "$#" -eq 3 ]; then
    echo "Usage: $0 path_to_fps2 path_to_extracted_fps2 [--cuda]"
    exit 1
fi

input=$1
out=$2


echo "0.00% - processing 02ae15c9-7d9c-4821-bc8f-10281332d69b--510-40"
mkdir -p $out/02ae15c9-7d9c-4821-bc8f-10281332d69b--510-40
for i in $(seq -f "%010g" 510	549	); do rsync -a $input/02ae15c9-7d9c-4821-bc8f-10281332d69b/img${i}.jpg $out/02ae15c9-7d9c-4821-bc8f-10281332d69b--510-40; done

echo "0.50% - processing 02ec195f-1841-402c-8a83-34f345d96a0c--3319-40"
mkdir -p $out/02ec195f-1841-402c-8a83-34f345d96a0c--3319-40
for i in $(seq -f "%010g" 3319	3358	); do rsync -a $input/02ec195f-1841-402c-8a83-34f345d96a0c/img${i}.jpg $out/02ec195f-1841-402c-8a83-34f345d96a0c--3319-40; done

echo "1.00% - processing 03a84301-1175-41f4-bed3-7a71fda30913--2549-20"
mkdir -p $out/03a84301-1175-41f4-bed3-7a71fda30913--2549-20
for i in $(seq -f "%010g" 2549	2568	); do rsync -a $input/03a84301-1175-41f4-bed3-7a71fda30913/img${i}.jpg $out/03a84301-1175-41f4-bed3-7a71fda30913--2549-20; done

echo "1.50% - processing 03f46720-8327-4215-ba56-5d739f6795ef--681-10"
mkdir -p $out/03f46720-8327-4215-ba56-5d739f6795ef--681-10
for i in $(seq -f "%010g" 681	690	); do rsync -a $input/03f46720-8327-4215-ba56-5d739f6795ef/img${i}.jpg $out/03f46720-8327-4215-ba56-5d739f6795ef--681-10; done

echo "2.00% - processing 045b9a27-b894-49be-9d88-7d9f09ec77ef--6264-40"
mkdir -p $out/045b9a27-b894-49be-9d88-7d9f09ec77ef--6264-40
for i in $(seq -f "%010g" 6264	6303	); do rsync -a $input/045b9a27-b894-49be-9d88-7d9f09ec77ef/img${i}.jpg $out/045b9a27-b894-49be-9d88-7d9f09ec77ef--6264-40; done

echo "2.50% - processing 04ccf527-e1dd-47ff-b8d0-4cec6e975391--1410-40"
mkdir -p $out/04ccf527-e1dd-47ff-b8d0-4cec6e975391--1410-40
for i in $(seq -f "%010g" 1410	1449	); do rsync -a $input/04ccf527-e1dd-47ff-b8d0-4cec6e975391/img${i}.jpg $out/04ccf527-e1dd-47ff-b8d0-4cec6e975391--1410-40; done

echo "3.00% - processing 0518d285-b7b0-4f98-ae06-a95e0248ccfc--3027-10"
mkdir -p $out/0518d285-b7b0-4f98-ae06-a95e0248ccfc--3027-10
for i in $(seq -f "%010g" 3027	3036	); do rsync -a $input/0518d285-b7b0-4f98-ae06-a95e0248ccfc/img${i}.jpg $out/0518d285-b7b0-4f98-ae06-a95e0248ccfc--3027-10; done

echo "3.50% - processing 0518d285-b7b0-4f98-ae06-a95e0248ccfc--3057-20"
mkdir -p $out/0518d285-b7b0-4f98-ae06-a95e0248ccfc--3057-20
for i in $(seq -f "%010g" 3057	3076	); do rsync -a $input/0518d285-b7b0-4f98-ae06-a95e0248ccfc/img${i}.jpg $out/0518d285-b7b0-4f98-ae06-a95e0248ccfc--3057-20; done

echo "4.00% - processing 057670b3-f243-416c-b41b-82d309fd2b88--226-40"
mkdir -p $out/057670b3-f243-416c-b41b-82d309fd2b88--226-40
for i in $(seq -f "%010g" 226	265	); do rsync -a $input/057670b3-f243-416c-b41b-82d309fd2b88/img${i}.jpg $out/057670b3-f243-416c-b41b-82d309fd2b88--226-40; done

echo "4.50% - processing 05b56572-491f-4fa4-8336-248383b6ce0c--849-40"
mkdir -p $out/05b56572-491f-4fa4-8336-248383b6ce0c--849-40
for i in $(seq -f "%010g" 849	888	); do rsync -a $input/05b56572-491f-4fa4-8336-248383b6ce0c/img${i}.jpg $out/05b56572-491f-4fa4-8336-248383b6ce0c--849-40; done

echo "5.00% - processing 06ca18e2-d71c-48f0-b21a-52e42206aabf--617-40"
mkdir -p $out/06ca18e2-d71c-48f0-b21a-52e42206aabf--617-40
for i in $(seq -f "%010g" 617	656	); do rsync -a $input/06ca18e2-d71c-48f0-b21a-52e42206aabf/img${i}.jpg $out/06ca18e2-d71c-48f0-b21a-52e42206aabf--617-40; done

echo "5.50% - processing 07287862-edf9-45d8-a33b-853943d3aae5--6343-30"
mkdir -p $out/07287862-edf9-45d8-a33b-853943d3aae5--6343-30
for i in $(seq -f "%010g" 6343	6372	); do rsync -a $input/07287862-edf9-45d8-a33b-853943d3aae5/img${i}.jpg $out/07287862-edf9-45d8-a33b-853943d3aae5--6343-30; done

echo "6.00% - processing 0869732c-5228-4a28-9c5a-467a5ebef78c--751-10"
mkdir -p $out/0869732c-5228-4a28-9c5a-467a5ebef78c--751-10
for i in $(seq -f "%010g" 751	760	); do rsync -a $input/0869732c-5228-4a28-9c5a-467a5ebef78c/img${i}.jpg $out/0869732c-5228-4a28-9c5a-467a5ebef78c--751-10; done

echo "6.50% - processing 0869732c-5228-4a28-9c5a-467a5ebef78c--793-20"
mkdir -p $out/0869732c-5228-4a28-9c5a-467a5ebef78c--793-20
for i in $(seq -f "%010g" 793	812	); do rsync -a $input/0869732c-5228-4a28-9c5a-467a5ebef78c/img${i}.jpg $out/0869732c-5228-4a28-9c5a-467a5ebef78c--793-20; done

echo "7.00% - processing 0869732c-5228-4a28-9c5a-467a5ebef78c--845-30"
mkdir -p $out/0869732c-5228-4a28-9c5a-467a5ebef78c--845-30
for i in $(seq -f "%010g" 845	874	); do rsync -a $input/0869732c-5228-4a28-9c5a-467a5ebef78c/img${i}.jpg $out/0869732c-5228-4a28-9c5a-467a5ebef78c--845-30; done

echo "7.50% - processing 08d94a5d-7c4b-4b1c-af12-c1ceb53c51c8--1256-40"
mkdir -p $out/08d94a5d-7c4b-4b1c-af12-c1ceb53c51c8--1256-40
for i in $(seq -f "%010g" 1256	1295	); do rsync -a $input/08d94a5d-7c4b-4b1c-af12-c1ceb53c51c8/img${i}.jpg $out/08d94a5d-7c4b-4b1c-af12-c1ceb53c51c8--1256-40; done

echo "8.00% - processing 090042fc-0539-4a5b-b644-f3f6c8201711--4995-40"
mkdir -p $out/090042fc-0539-4a5b-b644-f3f6c8201711--4995-40
for i in $(seq -f "%010g" 4995	5034	); do rsync -a $input/090042fc-0539-4a5b-b644-f3f6c8201711/img${i}.jpg $out/090042fc-0539-4a5b-b644-f3f6c8201711--4995-40; done

echo "8.50% - processing 09090dbb-3d4d-4f04-a6c6-d8215a0143f7--107-20"
mkdir -p $out/09090dbb-3d4d-4f04-a6c6-d8215a0143f7--107-20
for i in $(seq -f "%010g" 107	126	); do rsync -a $input/09090dbb-3d4d-4f04-a6c6-d8215a0143f7/img${i}.jpg $out/09090dbb-3d4d-4f04-a6c6-d8215a0143f7--107-20; done

echo "9.00% - processing 0959d7f3-0e58-486f-a21c-ec82b321469c--6747-30"
mkdir -p $out/0959d7f3-0e58-486f-a21c-ec82b321469c--6747-30
for i in $(seq -f "%010g" 6747	6776	); do rsync -a $input/0959d7f3-0e58-486f-a21c-ec82b321469c/img${i}.jpg $out/0959d7f3-0e58-486f-a21c-ec82b321469c--6747-30; done

echo "9.50% - processing 09f525f2-fff2-46ea-ab34-f1e1d64eec92--3045-30"
mkdir -p $out/09f525f2-fff2-46ea-ab34-f1e1d64eec92--3045-30
for i in $(seq -f "%010g" 3045	3074	); do rsync -a $input/09f525f2-fff2-46ea-ab34-f1e1d64eec92/img${i}.jpg $out/09f525f2-fff2-46ea-ab34-f1e1d64eec92--3045-30; done

echo "10.00% - processing 09f525f2-fff2-46ea-ab34-f1e1d64eec92--821-10"
mkdir -p $out/09f525f2-fff2-46ea-ab34-f1e1d64eec92--821-10
for i in $(seq -f "%010g" 821	830	); do rsync -a $input/09f525f2-fff2-46ea-ab34-f1e1d64eec92/img${i}.jpg $out/09f525f2-fff2-46ea-ab34-f1e1d64eec92--821-10; done

echo "10.50% - processing 09ff369a-a147-4ba9-bfea-499b7fbe554f--1329-10"
mkdir -p $out/09ff369a-a147-4ba9-bfea-499b7fbe554f--1329-10
for i in $(seq -f "%010g" 1329	1338	); do rsync -a $input/09ff369a-a147-4ba9-bfea-499b7fbe554f/img${i}.jpg $out/09ff369a-a147-4ba9-bfea-499b7fbe554f--1329-10; done

echo "11.00% - processing 09ff369a-a147-4ba9-bfea-499b7fbe554f--1435-30"
mkdir -p $out/09ff369a-a147-4ba9-bfea-499b7fbe554f--1435-30
for i in $(seq -f "%010g" 1435	1464	); do rsync -a $input/09ff369a-a147-4ba9-bfea-499b7fbe554f/img${i}.jpg $out/09ff369a-a147-4ba9-bfea-499b7fbe554f--1435-30; done

echo "11.50% - processing 09ff369a-a147-4ba9-bfea-499b7fbe554f--201-20"
mkdir -p $out/09ff369a-a147-4ba9-bfea-499b7fbe554f--201-20
for i in $(seq -f "%010g" 201	220	); do rsync -a $input/09ff369a-a147-4ba9-bfea-499b7fbe554f/img${i}.jpg $out/09ff369a-a147-4ba9-bfea-499b7fbe554f--201-20; done

echo "12.00% - processing 0a02a1ed-a327-4753-b270-e95298984b96--395-10"
mkdir -p $out/0a02a1ed-a327-4753-b270-e95298984b96--395-10
for i in $(seq -f "%010g" 395	404	); do rsync -a $input/0a02a1ed-a327-4753-b270-e95298984b96/img${i}.jpg $out/0a02a1ed-a327-4753-b270-e95298984b96--395-10; done

echo "12.50% - processing 0b2b5dee-272f-4eb5-b553-756f1786aae7--1217-30"
mkdir -p $out/0b2b5dee-272f-4eb5-b553-756f1786aae7--1217-30
for i in $(seq -f "%010g" 1217	1246	); do rsync -a $input/0b2b5dee-272f-4eb5-b553-756f1786aae7/img${i}.jpg $out/0b2b5dee-272f-4eb5-b553-756f1786aae7--1217-30; done

echo "13.00% - processing 0b2b5dee-272f-4eb5-b553-756f1786aae7--785-20"
mkdir -p $out/0b2b5dee-272f-4eb5-b553-756f1786aae7--785-20
for i in $(seq -f "%010g" 785	804	); do rsync -a $input/0b2b5dee-272f-4eb5-b553-756f1786aae7/img${i}.jpg $out/0b2b5dee-272f-4eb5-b553-756f1786aae7--785-20; done

echo "13.50% - processing 0b2e10d7-6d96-4a4d-a7f9-69656f3ac20b--9-30"
mkdir -p $out/0b2e10d7-6d96-4a4d-a7f9-69656f3ac20b--9-30
for i in $(seq -f "%010g" 9	38	); do rsync -a $input/0b2e10d7-6d96-4a4d-a7f9-69656f3ac20b/img${i}.jpg $out/0b2e10d7-6d96-4a4d-a7f9-69656f3ac20b--9-30; done

echo "14.00% - processing 0d23c547-019c-4079-9f3f-531598270143--1365-10"
mkdir -p $out/0d23c547-019c-4079-9f3f-531598270143--1365-10
for i in $(seq -f "%010g" 1365	1374	); do rsync -a $input/0d23c547-019c-4079-9f3f-531598270143/img${i}.jpg $out/0d23c547-019c-4079-9f3f-531598270143--1365-10; done

echo "14.50% - processing 0d23c547-019c-4079-9f3f-531598270143--1383-20"
mkdir -p $out/0d23c547-019c-4079-9f3f-531598270143--1383-20
for i in $(seq -f "%010g" 1383	1402	); do rsync -a $input/0d23c547-019c-4079-9f3f-531598270143/img${i}.jpg $out/0d23c547-019c-4079-9f3f-531598270143--1383-20; done

echo "15.00% - processing 0d23c547-019c-4079-9f3f-531598270143--833-30"
mkdir -p $out/0d23c547-019c-4079-9f3f-531598270143--833-30
for i in $(seq -f "%010g" 833	862	); do rsync -a $input/0d23c547-019c-4079-9f3f-531598270143/img${i}.jpg $out/0d23c547-019c-4079-9f3f-531598270143--833-30; done

echo "15.50% - processing 0d32d3bd-bfc8-4631-9002-7ffd97e4d65f--1127-10"
mkdir -p $out/0d32d3bd-bfc8-4631-9002-7ffd97e4d65f--1127-10
for i in $(seq -f "%010g" 1127	1136	); do rsync -a $input/0d32d3bd-bfc8-4631-9002-7ffd97e4d65f/img${i}.jpg $out/0d32d3bd-bfc8-4631-9002-7ffd97e4d65f--1127-10; done

echo "16.00% - processing 0d32d3bd-bfc8-4631-9002-7ffd97e4d65f--1243-20"
mkdir -p $out/0d32d3bd-bfc8-4631-9002-7ffd97e4d65f--1243-20
for i in $(seq -f "%010g" 1243	1262	); do rsync -a $input/0d32d3bd-bfc8-4631-9002-7ffd97e4d65f/img${i}.jpg $out/0d32d3bd-bfc8-4631-9002-7ffd97e4d65f--1243-20; done

echo "16.50% - processing 0d32d3bd-bfc8-4631-9002-7ffd97e4d65f--663-30"
mkdir -p $out/0d32d3bd-bfc8-4631-9002-7ffd97e4d65f--663-30
for i in $(seq -f "%010g" 663	692	); do rsync -a $input/0d32d3bd-bfc8-4631-9002-7ffd97e4d65f/img${i}.jpg $out/0d32d3bd-bfc8-4631-9002-7ffd97e4d65f--663-30; done

echo "17.00% - processing 0d9ae668-5e94-48b8-b5f0-77742ab9f2cf--229-20"
mkdir -p $out/0d9ae668-5e94-48b8-b5f0-77742ab9f2cf--229-20
for i in $(seq -f "%010g" 229	248	); do rsync -a $input/0d9ae668-5e94-48b8-b5f0-77742ab9f2cf/img${i}.jpg $out/0d9ae668-5e94-48b8-b5f0-77742ab9f2cf--229-20; done

echo "17.50% - processing 0f1446cc-b006-46a7-9f47-66a51ac5ac5b--1088-40"
mkdir -p $out/0f1446cc-b006-46a7-9f47-66a51ac5ac5b--1088-40
for i in $(seq -f "%010g" 1088	1127	); do rsync -a $input/0f1446cc-b006-46a7-9f47-66a51ac5ac5b/img${i}.jpg $out/0f1446cc-b006-46a7-9f47-66a51ac5ac5b--1088-40; done

echo "18.00% - processing 0f74f1ba-7223-4990-8fb8-41df61742988--1203-30"
mkdir -p $out/0f74f1ba-7223-4990-8fb8-41df61742988--1203-30
for i in $(seq -f "%010g" 1203	1232	); do rsync -a $input/0f74f1ba-7223-4990-8fb8-41df61742988/img${i}.jpg $out/0f74f1ba-7223-4990-8fb8-41df61742988--1203-30; done

echo "18.50% - processing 0f74f1ba-7223-4990-8fb8-41df61742988--821-20"
mkdir -p $out/0f74f1ba-7223-4990-8fb8-41df61742988--821-20
for i in $(seq -f "%010g" 821	840	); do rsync -a $input/0f74f1ba-7223-4990-8fb8-41df61742988/img${i}.jpg $out/0f74f1ba-7223-4990-8fb8-41df61742988--821-20; done

echo "19.00% - processing 0f74f1ba-7223-4990-8fb8-41df61742988--965-10"
mkdir -p $out/0f74f1ba-7223-4990-8fb8-41df61742988--965-10
for i in $(seq -f "%010g" 965	974	); do rsync -a $input/0f74f1ba-7223-4990-8fb8-41df61742988/img${i}.jpg $out/0f74f1ba-7223-4990-8fb8-41df61742988--965-10; done

echo "19.50% - processing 0fb8012d-430d-4b45-a8eb-5fd8cb43f5b0--6145-30"
mkdir -p $out/0fb8012d-430d-4b45-a8eb-5fd8cb43f5b0--6145-30
for i in $(seq -f "%010g" 6145	6174	); do rsync -a $input/0fb8012d-430d-4b45-a8eb-5fd8cb43f5b0/img${i}.jpg $out/0fb8012d-430d-4b45-a8eb-5fd8cb43f5b0--6145-30; done

echo "20.00% - processing 0fb8012d-430d-4b45-a8eb-5fd8cb43f5b0--6703-20"
mkdir -p $out/0fb8012d-430d-4b45-a8eb-5fd8cb43f5b0--6703-20
for i in $(seq -f "%010g" 6703	6722	); do rsync -a $input/0fb8012d-430d-4b45-a8eb-5fd8cb43f5b0/img${i}.jpg $out/0fb8012d-430d-4b45-a8eb-5fd8cb43f5b0--6703-20; done

echo "20.50% - processing 0fb8012d-430d-4b45-a8eb-5fd8cb43f5b0--7635-10"
mkdir -p $out/0fb8012d-430d-4b45-a8eb-5fd8cb43f5b0--7635-10
for i in $(seq -f "%010g" 7635	7644	); do rsync -a $input/0fb8012d-430d-4b45-a8eb-5fd8cb43f5b0/img${i}.jpg $out/0fb8012d-430d-4b45-a8eb-5fd8cb43f5b0--7635-10; done

echo "21.00% - processing 10358e57-51f9-4db5-903e-11529bdde9a6--477-20"
mkdir -p $out/10358e57-51f9-4db5-903e-11529bdde9a6--477-20
for i in $(seq -f "%010g" 477	496	); do rsync -a $input/10358e57-51f9-4db5-903e-11529bdde9a6/img${i}.jpg $out/10358e57-51f9-4db5-903e-11529bdde9a6--477-20; done

echo "21.50% - processing 10358e57-51f9-4db5-903e-11529bdde9a6--607-10"
mkdir -p $out/10358e57-51f9-4db5-903e-11529bdde9a6--607-10
for i in $(seq -f "%010g" 607	616	); do rsync -a $input/10358e57-51f9-4db5-903e-11529bdde9a6/img${i}.jpg $out/10358e57-51f9-4db5-903e-11529bdde9a6--607-10; done

echo "22.00% - processing 1074214d-81aa-4dcf-865b-29816133d26c--1447-30"
mkdir -p $out/1074214d-81aa-4dcf-865b-29816133d26c--1447-30
for i in $(seq -f "%010g" 1447	1476	); do rsync -a $input/1074214d-81aa-4dcf-865b-29816133d26c/img${i}.jpg $out/1074214d-81aa-4dcf-865b-29816133d26c--1447-30; done

echo "22.50% - processing 1074214d-81aa-4dcf-865b-29816133d26c--2343-10"
mkdir -p $out/1074214d-81aa-4dcf-865b-29816133d26c--2343-10
for i in $(seq -f "%010g" 2343	2352	); do rsync -a $input/1074214d-81aa-4dcf-865b-29816133d26c/img${i}.jpg $out/1074214d-81aa-4dcf-865b-29816133d26c--2343-10; done

echo "23.00% - processing 108c50a3-aa31-4df9-9915-78b16f000d5b--381-30"
mkdir -p $out/108c50a3-aa31-4df9-9915-78b16f000d5b--381-30
for i in $(seq -f "%010g" 381	410	); do rsync -a $input/108c50a3-aa31-4df9-9915-78b16f000d5b/img${i}.jpg $out/108c50a3-aa31-4df9-9915-78b16f000d5b--381-30; done

echo "23.50% - processing 12305d5a-55d1-433a-8b60-2a2e85f1c85e--142-40"
mkdir -p $out/12305d5a-55d1-433a-8b60-2a2e85f1c85e--142-40
for i in $(seq -f "%010g" 142	181	); do rsync -a $input/12305d5a-55d1-433a-8b60-2a2e85f1c85e/img${i}.jpg $out/12305d5a-55d1-433a-8b60-2a2e85f1c85e--142-40; done

echo "24.00% - processing 1237a78f-3a97-48e6-8ffc-81b49b64fc40--1205-20"
mkdir -p $out/1237a78f-3a97-48e6-8ffc-81b49b64fc40--1205-20
for i in $(seq -f "%010g" 1205	1224	); do rsync -a $input/1237a78f-3a97-48e6-8ffc-81b49b64fc40/img${i}.jpg $out/1237a78f-3a97-48e6-8ffc-81b49b64fc40--1205-20; done

echo "24.50% - processing 1237a78f-3a97-48e6-8ffc-81b49b64fc40--1245-30"
mkdir -p $out/1237a78f-3a97-48e6-8ffc-81b49b64fc40--1245-30
for i in $(seq -f "%010g" 1245	1274	); do rsync -a $input/1237a78f-3a97-48e6-8ffc-81b49b64fc40/img${i}.jpg $out/1237a78f-3a97-48e6-8ffc-81b49b64fc40--1245-30; done

echo "25.00% - processing 1286d04f-fe64-45f9-9253-de126f61164e--1713-30"
mkdir -p $out/1286d04f-fe64-45f9-9253-de126f61164e--1713-30
for i in $(seq -f "%010g" 1713	1742	); do rsync -a $input/1286d04f-fe64-45f9-9253-de126f61164e/img${i}.jpg $out/1286d04f-fe64-45f9-9253-de126f61164e--1713-30; done

echo "25.50% - processing 13e513d6-fa5a-4909-9acf-3f29d022ef42--959-40"
mkdir -p $out/13e513d6-fa5a-4909-9acf-3f29d022ef42--959-40
for i in $(seq -f "%010g" 959	998	); do rsync -a $input/13e513d6-fa5a-4909-9acf-3f29d022ef42/img${i}.jpg $out/13e513d6-fa5a-4909-9acf-3f29d022ef42--959-40; done

echo "26.00% - processing 14177ca9-924f-4e68-b218-6009dbddc7cc--175-10"
mkdir -p $out/14177ca9-924f-4e68-b218-6009dbddc7cc--175-10
for i in $(seq -f "%010g" 175	184	); do rsync -a $input/14177ca9-924f-4e68-b218-6009dbddc7cc/img${i}.jpg $out/14177ca9-924f-4e68-b218-6009dbddc7cc--175-10; done

echo "26.50% - processing 14177ca9-924f-4e68-b218-6009dbddc7cc--333-20"
mkdir -p $out/14177ca9-924f-4e68-b218-6009dbddc7cc--333-20
for i in $(seq -f "%010g" 333	352	); do rsync -a $input/14177ca9-924f-4e68-b218-6009dbddc7cc/img${i}.jpg $out/14177ca9-924f-4e68-b218-6009dbddc7cc--333-20; done

echo "27.00% - processing 14571685-d82b-4fda-907f-1165aa9fd9ea--113-20"
mkdir -p $out/14571685-d82b-4fda-907f-1165aa9fd9ea--113-20
for i in $(seq -f "%010g" 113	132	); do rsync -a $input/14571685-d82b-4fda-907f-1165aa9fd9ea/img${i}.jpg $out/14571685-d82b-4fda-907f-1165aa9fd9ea--113-20; done

echo "27.50% - processing 150c104e-f9bd-4a81-b885-0e8b9968cada--2149-20"
mkdir -p $out/150c104e-f9bd-4a81-b885-0e8b9968cada--2149-20
for i in $(seq -f "%010g" 2149	2168	); do rsync -a $input/150c104e-f9bd-4a81-b885-0e8b9968cada/img${i}.jpg $out/150c104e-f9bd-4a81-b885-0e8b9968cada--2149-20; done

echo "28.00% - processing 150c104e-f9bd-4a81-b885-0e8b9968cada--859-30"
mkdir -p $out/150c104e-f9bd-4a81-b885-0e8b9968cada--859-30
for i in $(seq -f "%010g" 859	888	); do rsync -a $input/150c104e-f9bd-4a81-b885-0e8b9968cada/img${i}.jpg $out/150c104e-f9bd-4a81-b885-0e8b9968cada--859-30; done

echo "28.50% - processing 15497cd8-1eac-4926-aa06-ff07f33cc2a8--633-20"
mkdir -p $out/15497cd8-1eac-4926-aa06-ff07f33cc2a8--633-20
for i in $(seq -f "%010g" 633	652	); do rsync -a $input/15497cd8-1eac-4926-aa06-ff07f33cc2a8/img${i}.jpg $out/15497cd8-1eac-4926-aa06-ff07f33cc2a8--633-20; done

echo "29.00% - processing 16d4e2b1-dab0-40d4-9d44-c9e1c94b18a0--2815-40"
mkdir -p $out/16d4e2b1-dab0-40d4-9d44-c9e1c94b18a0--2815-40
for i in $(seq -f "%010g" 2815	2854	); do rsync -a $input/16d4e2b1-dab0-40d4-9d44-c9e1c94b18a0/img${i}.jpg $out/16d4e2b1-dab0-40d4-9d44-c9e1c94b18a0--2815-40; done

echo "29.50% - processing 17607d8a-806f-4990-b5e7-756b6389c826--1131-40"
mkdir -p $out/17607d8a-806f-4990-b5e7-756b6389c826--1131-40
for i in $(seq -f "%010g" 1131	1170	); do rsync -a $input/17607d8a-806f-4990-b5e7-756b6389c826/img${i}.jpg $out/17607d8a-806f-4990-b5e7-756b6389c826--1131-40; done

echo "30.00% - processing 17f2e8c9-ace4-48ff-9ef9-643715905be0--1551-10"
mkdir -p $out/17f2e8c9-ace4-48ff-9ef9-643715905be0--1551-10
for i in $(seq -f "%010g" 1551	1560	); do rsync -a $input/17f2e8c9-ace4-48ff-9ef9-643715905be0/img${i}.jpg $out/17f2e8c9-ace4-48ff-9ef9-643715905be0--1551-10; done

echo "30.50% - processing 17f2e8c9-ace4-48ff-9ef9-643715905be0--39-20"
mkdir -p $out/17f2e8c9-ace4-48ff-9ef9-643715905be0--39-20
for i in $(seq -f "%010g" 39	58	); do rsync -a $input/17f2e8c9-ace4-48ff-9ef9-643715905be0/img${i}.jpg $out/17f2e8c9-ace4-48ff-9ef9-643715905be0--39-20; done

echo "31.00% - processing 18e7a9c5-3a01-4956-a21b-ff5dcd8d9ba3--27-30"
mkdir -p $out/18e7a9c5-3a01-4956-a21b-ff5dcd8d9ba3--27-30
for i in $(seq -f "%010g" 27	56	); do rsync -a $input/18e7a9c5-3a01-4956-a21b-ff5dcd8d9ba3/img${i}.jpg $out/18e7a9c5-3a01-4956-a21b-ff5dcd8d9ba3--27-30; done

echo "31.50% - processing 18e7a9c5-3a01-4956-a21b-ff5dcd8d9ba3--2763-10"
mkdir -p $out/18e7a9c5-3a01-4956-a21b-ff5dcd8d9ba3--2763-10
for i in $(seq -f "%010g" 2763	2772	); do rsync -a $input/18e7a9c5-3a01-4956-a21b-ff5dcd8d9ba3/img${i}.jpg $out/18e7a9c5-3a01-4956-a21b-ff5dcd8d9ba3--2763-10; done

echo "32.00% - processing 18e7a9c5-3a01-4956-a21b-ff5dcd8d9ba3--3659-20"
mkdir -p $out/18e7a9c5-3a01-4956-a21b-ff5dcd8d9ba3--3659-20
for i in $(seq -f "%010g" 3659	3678	); do rsync -a $input/18e7a9c5-3a01-4956-a21b-ff5dcd8d9ba3/img${i}.jpg $out/18e7a9c5-3a01-4956-a21b-ff5dcd8d9ba3--3659-20; done

echo "32.50% - processing 1920cc38-ed21-4e7b-86dd-2a482c8ca321--1707-20"
mkdir -p $out/1920cc38-ed21-4e7b-86dd-2a482c8ca321--1707-20
for i in $(seq -f "%010g" 1707	1726	); do rsync -a $input/1920cc38-ed21-4e7b-86dd-2a482c8ca321/img${i}.jpg $out/1920cc38-ed21-4e7b-86dd-2a482c8ca321--1707-20; done

echo "33.00% - processing 1920cc38-ed21-4e7b-86dd-2a482c8ca321--2939-10"
mkdir -p $out/1920cc38-ed21-4e7b-86dd-2a482c8ca321--2939-10
for i in $(seq -f "%010g" 2939	2948	); do rsync -a $input/1920cc38-ed21-4e7b-86dd-2a482c8ca321/img${i}.jpg $out/1920cc38-ed21-4e7b-86dd-2a482c8ca321--2939-10; done

echo "33.50% - processing 1920cc38-ed21-4e7b-86dd-2a482c8ca321--3581-30"
mkdir -p $out/1920cc38-ed21-4e7b-86dd-2a482c8ca321--3581-30
for i in $(seq -f "%010g" 3581	3610	); do rsync -a $input/1920cc38-ed21-4e7b-86dd-2a482c8ca321/img${i}.jpg $out/1920cc38-ed21-4e7b-86dd-2a482c8ca321--3581-30; done

echo "34.00% - processing 1938c632-f575-49dd-8ae0-e48dbb467920--1321-20"
mkdir -p $out/1938c632-f575-49dd-8ae0-e48dbb467920--1321-20
for i in $(seq -f "%010g" 1321	1340	); do rsync -a $input/1938c632-f575-49dd-8ae0-e48dbb467920/img${i}.jpg $out/1938c632-f575-49dd-8ae0-e48dbb467920--1321-20; done

echo "34.50% - processing 1938c632-f575-49dd-8ae0-e48dbb467920--839-30"
mkdir -p $out/1938c632-f575-49dd-8ae0-e48dbb467920--839-30
for i in $(seq -f "%010g" 839	868	); do rsync -a $input/1938c632-f575-49dd-8ae0-e48dbb467920/img${i}.jpg $out/1938c632-f575-49dd-8ae0-e48dbb467920--839-30; done

echo "35.00% - processing 196e0e8c-f29f-48de-8e1e-ce52c2e76641--2901-10"
mkdir -p $out/196e0e8c-f29f-48de-8e1e-ce52c2e76641--2901-10
for i in $(seq -f "%010g" 2901	2910	); do rsync -a $input/196e0e8c-f29f-48de-8e1e-ce52c2e76641/img${i}.jpg $out/196e0e8c-f29f-48de-8e1e-ce52c2e76641--2901-10; done

echo "35.50% - processing 196e0e8c-f29f-48de-8e1e-ce52c2e76641--4881-20"
mkdir -p $out/196e0e8c-f29f-48de-8e1e-ce52c2e76641--4881-20
for i in $(seq -f "%010g" 4881	4900	); do rsync -a $input/196e0e8c-f29f-48de-8e1e-ce52c2e76641/img${i}.jpg $out/196e0e8c-f29f-48de-8e1e-ce52c2e76641--4881-20; done

echo "36.00% - processing 196e0e8c-f29f-48de-8e1e-ce52c2e76641--5335-30"
mkdir -p $out/196e0e8c-f29f-48de-8e1e-ce52c2e76641--5335-30
for i in $(seq -f "%010g" 5335	5364	); do rsync -a $input/196e0e8c-f29f-48de-8e1e-ce52c2e76641/img${i}.jpg $out/196e0e8c-f29f-48de-8e1e-ce52c2e76641--5335-30; done

echo "36.50% - processing 19a49df1-db3f-45d7-92f4-d965ee20edb7--617-10"
mkdir -p $out/19a49df1-db3f-45d7-92f4-d965ee20edb7--617-10
for i in $(seq -f "%010g" 617	626	); do rsync -a $input/19a49df1-db3f-45d7-92f4-d965ee20edb7/img${i}.jpg $out/19a49df1-db3f-45d7-92f4-d965ee20edb7--617-10; done

echo "37.00% - processing 19a49df1-db3f-45d7-92f4-d965ee20edb7--895-20"
mkdir -p $out/19a49df1-db3f-45d7-92f4-d965ee20edb7--895-20
for i in $(seq -f "%010g" 895	914	); do rsync -a $input/19a49df1-db3f-45d7-92f4-d965ee20edb7/img${i}.jpg $out/19a49df1-db3f-45d7-92f4-d965ee20edb7--895-20; done

echo "37.50% - processing 19fb5546-43a2-4a7b-9afd-672e7fd257c4--573-10"
mkdir -p $out/19fb5546-43a2-4a7b-9afd-672e7fd257c4--573-10
for i in $(seq -f "%010g" 573	582	); do rsync -a $input/19fb5546-43a2-4a7b-9afd-672e7fd257c4/img${i}.jpg $out/19fb5546-43a2-4a7b-9afd-672e7fd257c4--573-10; done

echo "38.00% - processing 19fb5546-43a2-4a7b-9afd-672e7fd257c4--807-20"
mkdir -p $out/19fb5546-43a2-4a7b-9afd-672e7fd257c4--807-20
for i in $(seq -f "%010g" 807	826	); do rsync -a $input/19fb5546-43a2-4a7b-9afd-672e7fd257c4/img${i}.jpg $out/19fb5546-43a2-4a7b-9afd-672e7fd257c4--807-20; done

echo "38.50% - processing 1a651dee-f65c-4005-829b-4824377929cc--295-20"
mkdir -p $out/1a651dee-f65c-4005-829b-4824377929cc--295-20
for i in $(seq -f "%010g" 295	314	); do rsync -a $input/1a651dee-f65c-4005-829b-4824377929cc/img${i}.jpg $out/1a651dee-f65c-4005-829b-4824377929cc--295-20; done

echo "39.00% - processing 1a651dee-f65c-4005-829b-4824377929cc--391-10"
mkdir -p $out/1a651dee-f65c-4005-829b-4824377929cc--391-10
for i in $(seq -f "%010g" 391	400	); do rsync -a $input/1a651dee-f65c-4005-829b-4824377929cc/img${i}.jpg $out/1a651dee-f65c-4005-829b-4824377929cc--391-10; done

echo "39.50% - processing 1a9bb53f-01ae-4bd7-afa3-a92570679a7a--65-10"
mkdir -p $out/1a9bb53f-01ae-4bd7-afa3-a92570679a7a--65-10
for i in $(seq -f "%010g" 65	74	); do rsync -a $input/1a9bb53f-01ae-4bd7-afa3-a92570679a7a/img${i}.jpg $out/1a9bb53f-01ae-4bd7-afa3-a92570679a7a--65-10; done

echo "40.00% - processing 1aa28dec-e81d-4a76-b808-b804efaae529--45-20"
mkdir -p $out/1aa28dec-e81d-4a76-b808-b804efaae529--45-20
for i in $(seq -f "%010g" 45	64	); do rsync -a $input/1aa28dec-e81d-4a76-b808-b804efaae529/img${i}.jpg $out/1aa28dec-e81d-4a76-b808-b804efaae529--45-20; done

echo "40.50% - processing 1b2f832a-7c17-4ca8-b7c3-49ac9dbaceb3--339-30"
mkdir -p $out/1b2f832a-7c17-4ca8-b7c3-49ac9dbaceb3--339-30
for i in $(seq -f "%010g" 339	368	); do rsync -a $input/1b2f832a-7c17-4ca8-b7c3-49ac9dbaceb3/img${i}.jpg $out/1b2f832a-7c17-4ca8-b7c3-49ac9dbaceb3--339-30; done

echo "41.00% - processing 1b9c07d3-0656-48ad-bae4-cca25f6664c9--2156-40"
mkdir -p $out/1b9c07d3-0656-48ad-bae4-cca25f6664c9--2156-40
for i in $(seq -f "%010g" 2156	2195	); do rsync -a $input/1b9c07d3-0656-48ad-bae4-cca25f6664c9/img${i}.jpg $out/1b9c07d3-0656-48ad-bae4-cca25f6664c9--2156-40; done

echo "41.50% - processing 1c14c094-8a86-4e61-bf94-5e3ad7cbd120--2002-40"
mkdir -p $out/1c14c094-8a86-4e61-bf94-5e3ad7cbd120--2002-40
for i in $(seq -f "%010g" 2002	2041	); do rsync -a $input/1c14c094-8a86-4e61-bf94-5e3ad7cbd120/img${i}.jpg $out/1c14c094-8a86-4e61-bf94-5e3ad7cbd120--2002-40; done

echo "42.00% - processing 1c663b98-e28a-4036-b66e-17d942902d32--2156-40"
mkdir -p $out/1c663b98-e28a-4036-b66e-17d942902d32--2156-40
for i in $(seq -f "%010g" 2156	2195	); do rsync -a $input/1c663b98-e28a-4036-b66e-17d942902d32/img${i}.jpg $out/1c663b98-e28a-4036-b66e-17d942902d32--2156-40; done

echo "42.50% - processing 1da50887-dd33-4644-9377-1bf855b86ffd--981-20"
mkdir -p $out/1da50887-dd33-4644-9377-1bf855b86ffd--981-20
for i in $(seq -f "%010g" 981	1000	); do rsync -a $input/1da50887-dd33-4644-9377-1bf855b86ffd/img${i}.jpg $out/1da50887-dd33-4644-9377-1bf855b86ffd--981-20; done

echo "43.00% - processing 1df0665a-91f7-48ff-b74d-0e4b1c3729e0--483-20"
mkdir -p $out/1df0665a-91f7-48ff-b74d-0e4b1c3729e0--483-20
for i in $(seq -f "%010g" 483	502	); do rsync -a $input/1df0665a-91f7-48ff-b74d-0e4b1c3729e0/img${i}.jpg $out/1df0665a-91f7-48ff-b74d-0e4b1c3729e0--483-20; done

echo "43.50% - processing 1f4f8bf2-3fa9-4544-bf6b-baf5b9568efd--1247-20"
mkdir -p $out/1f4f8bf2-3fa9-4544-bf6b-baf5b9568efd--1247-20
for i in $(seq -f "%010g" 1247	1266	); do rsync -a $input/1f4f8bf2-3fa9-4544-bf6b-baf5b9568efd/img${i}.jpg $out/1f4f8bf2-3fa9-4544-bf6b-baf5b9568efd--1247-20; done

echo "44.00% - processing 1f4f8bf2-3fa9-4544-bf6b-baf5b9568efd--129-30"
mkdir -p $out/1f4f8bf2-3fa9-4544-bf6b-baf5b9568efd--129-30
for i in $(seq -f "%010g" 129	158	); do rsync -a $input/1f4f8bf2-3fa9-4544-bf6b-baf5b9568efd/img${i}.jpg $out/1f4f8bf2-3fa9-4544-bf6b-baf5b9568efd--129-30; done

echo "44.50% - processing 1f876f34-cf5e-4c85-bb28-a7f01be8d45d--1965-40"
mkdir -p $out/1f876f34-cf5e-4c85-bb28-a7f01be8d45d--1965-40
for i in $(seq -f "%010g" 1965	2004	); do rsync -a $input/1f876f34-cf5e-4c85-bb28-a7f01be8d45d/img${i}.jpg $out/1f876f34-cf5e-4c85-bb28-a7f01be8d45d--1965-40; done

echo "45.00% - processing 1f8b71aa-9e83-4e1e-8290-6be4c2e640d3--1-20"
mkdir -p $out/1f8b71aa-9e83-4e1e-8290-6be4c2e640d3--1-20
for i in $(seq -f "%010g" 1	20	); do rsync -a $input/1f8b71aa-9e83-4e1e-8290-6be4c2e640d3/img${i}.jpg $out/1f8b71aa-9e83-4e1e-8290-6be4c2e640d3--1-20; done

echo "45.50% - processing 1f8b71aa-9e83-4e1e-8290-6be4c2e640d3--493-10"
mkdir -p $out/1f8b71aa-9e83-4e1e-8290-6be4c2e640d3--493-10
for i in $(seq -f "%010g" 493	502	); do rsync -a $input/1f8b71aa-9e83-4e1e-8290-6be4c2e640d3/img${i}.jpg $out/1f8b71aa-9e83-4e1e-8290-6be4c2e640d3--493-10; done

echo "46.00% - processing 20d62117-8b5f-44b3-b86f-a936e6cdd2c2--1471-40"
mkdir -p $out/20d62117-8b5f-44b3-b86f-a936e6cdd2c2--1471-40
for i in $(seq -f "%010g" 1471	1510	); do rsync -a $input/20d62117-8b5f-44b3-b86f-a936e6cdd2c2/img${i}.jpg $out/20d62117-8b5f-44b3-b86f-a936e6cdd2c2--1471-40; done

echo "46.50% - processing 21339a6a-ea8b-41c9-b5df-b2e3529b6c25--1407-20"
mkdir -p $out/21339a6a-ea8b-41c9-b5df-b2e3529b6c25--1407-20
for i in $(seq -f "%010g" 1407	1426	); do rsync -a $input/21339a6a-ea8b-41c9-b5df-b2e3529b6c25/img${i}.jpg $out/21339a6a-ea8b-41c9-b5df-b2e3529b6c25--1407-20; done

echo "47.00% - processing 21339a6a-ea8b-41c9-b5df-b2e3529b6c25--1431-10"
mkdir -p $out/21339a6a-ea8b-41c9-b5df-b2e3529b6c25--1431-10
for i in $(seq -f "%010g" 1431	1440	); do rsync -a $input/21339a6a-ea8b-41c9-b5df-b2e3529b6c25/img${i}.jpg $out/21339a6a-ea8b-41c9-b5df-b2e3529b6c25--1431-10; done

echo "47.50% - processing 21d4012a-bc29-46ac-a2d0-d2ba504a2f83--287-10"
mkdir -p $out/21d4012a-bc29-46ac-a2d0-d2ba504a2f83--287-10
for i in $(seq -f "%010g" 287	296	); do rsync -a $input/21d4012a-bc29-46ac-a2d0-d2ba504a2f83/img${i}.jpg $out/21d4012a-bc29-46ac-a2d0-d2ba504a2f83--287-10; done

echo "48.00% - processing 21d4012a-bc29-46ac-a2d0-d2ba504a2f83--299-20"
mkdir -p $out/21d4012a-bc29-46ac-a2d0-d2ba504a2f83--299-20
for i in $(seq -f "%010g" 299	318	); do rsync -a $input/21d4012a-bc29-46ac-a2d0-d2ba504a2f83/img${i}.jpg $out/21d4012a-bc29-46ac-a2d0-d2ba504a2f83--299-20; done

echo "48.50% - processing 21d4012a-bc29-46ac-a2d0-d2ba504a2f83--85-30"
mkdir -p $out/21d4012a-bc29-46ac-a2d0-d2ba504a2f83--85-30
for i in $(seq -f "%010g" 85	114	); do rsync -a $input/21d4012a-bc29-46ac-a2d0-d2ba504a2f83/img${i}.jpg $out/21d4012a-bc29-46ac-a2d0-d2ba504a2f83--85-30; done

echo "49.00% - processing 227165a6-277b-451c-bcda-1d2cf60577ab--117-20"
mkdir -p $out/227165a6-277b-451c-bcda-1d2cf60577ab--117-20
for i in $(seq -f "%010g" 117	136	); do rsync -a $input/227165a6-277b-451c-bcda-1d2cf60577ab/img${i}.jpg $out/227165a6-277b-451c-bcda-1d2cf60577ab--117-20; done

echo "49.50% - processing 227165a6-277b-451c-bcda-1d2cf60577ab--843-10"
mkdir -p $out/227165a6-277b-451c-bcda-1d2cf60577ab--843-10
for i in $(seq -f "%010g" 843	852	); do rsync -a $input/227165a6-277b-451c-bcda-1d2cf60577ab/img${i}.jpg $out/227165a6-277b-451c-bcda-1d2cf60577ab--843-10; done

echo "50.00% - processing 229e8124-605e-4c86-b4ac-fc03e0a71111--321-20"
mkdir -p $out/229e8124-605e-4c86-b4ac-fc03e0a71111--321-20
for i in $(seq -f "%010g" 321	340	); do rsync -a $input/229e8124-605e-4c86-b4ac-fc03e0a71111/img${i}.jpg $out/229e8124-605e-4c86-b4ac-fc03e0a71111--321-20; done

echo "50.50% - processing 229e8124-605e-4c86-b4ac-fc03e0a71111--827-10"
mkdir -p $out/229e8124-605e-4c86-b4ac-fc03e0a71111--827-10
for i in $(seq -f "%010g" 827	836	); do rsync -a $input/229e8124-605e-4c86-b4ac-fc03e0a71111/img${i}.jpg $out/229e8124-605e-4c86-b4ac-fc03e0a71111--827-10; done

echo "51.00% - processing 22d198c2-4057-47b7-978f-90a1599c5980--203-30"
mkdir -p $out/22d198c2-4057-47b7-978f-90a1599c5980--203-30
for i in $(seq -f "%010g" 203	232	); do rsync -a $input/22d198c2-4057-47b7-978f-90a1599c5980/img${i}.jpg $out/22d198c2-4057-47b7-978f-90a1599c5980--203-30; done

echo "51.50% - processing 22d198c2-4057-47b7-978f-90a1599c5980--2995-10"
mkdir -p $out/22d198c2-4057-47b7-978f-90a1599c5980--2995-10
for i in $(seq -f "%010g" 2995	3004	); do rsync -a $input/22d198c2-4057-47b7-978f-90a1599c5980/img${i}.jpg $out/22d198c2-4057-47b7-978f-90a1599c5980--2995-10; done

echo "52.00% - processing 238c275a-c8ce-4429-922e-4dceabe09d29--295-10"
mkdir -p $out/238c275a-c8ce-4429-922e-4dceabe09d29--295-10
for i in $(seq -f "%010g" 295	304	); do rsync -a $input/238c275a-c8ce-4429-922e-4dceabe09d29/img${i}.jpg $out/238c275a-c8ce-4429-922e-4dceabe09d29--295-10; done

echo "52.50% - processing 238c275a-c8ce-4429-922e-4dceabe09d29--327-20"
mkdir -p $out/238c275a-c8ce-4429-922e-4dceabe09d29--327-20
for i in $(seq -f "%010g" 327	346	); do rsync -a $input/238c275a-c8ce-4429-922e-4dceabe09d29/img${i}.jpg $out/238c275a-c8ce-4429-922e-4dceabe09d29--327-20; done

echo "53.00% - processing 239e6608-8c7c-4643-9a64-edc7f3da0e8e--75-20"
mkdir -p $out/239e6608-8c7c-4643-9a64-edc7f3da0e8e--75-20
for i in $(seq -f "%010g" 75	94	); do rsync -a $input/239e6608-8c7c-4643-9a64-edc7f3da0e8e/img${i}.jpg $out/239e6608-8c7c-4643-9a64-edc7f3da0e8e--75-20; done

echo "53.50% - processing 23d27284-5421-4913-baba-b660fadd1920--1-40"
mkdir -p $out/23d27284-5421-4913-baba-b660fadd1920--1-40
for i in $(seq -f "%010g" 1	40	); do rsync -a $input/23d27284-5421-4913-baba-b660fadd1920/img${i}.jpg $out/23d27284-5421-4913-baba-b660fadd1920--1-40; done

echo "54.00% - processing 23e0af07-6932-4ccf-8551-3a720d141b89--1935-10"
mkdir -p $out/23e0af07-6932-4ccf-8551-3a720d141b89--1935-10
for i in $(seq -f "%010g" 1935	1944	); do rsync -a $input/23e0af07-6932-4ccf-8551-3a720d141b89/img${i}.jpg $out/23e0af07-6932-4ccf-8551-3a720d141b89--1935-10; done

echo "54.50% - processing 23e0af07-6932-4ccf-8551-3a720d141b89--4125-30"
mkdir -p $out/23e0af07-6932-4ccf-8551-3a720d141b89--4125-30
for i in $(seq -f "%010g" 4125	4154	); do rsync -a $input/23e0af07-6932-4ccf-8551-3a720d141b89/img${i}.jpg $out/23e0af07-6932-4ccf-8551-3a720d141b89--4125-30; done

echo "55.00% - processing 25b2f1d5-7266-401c-bdfa-32944607f564--375-20"
mkdir -p $out/25b2f1d5-7266-401c-bdfa-32944607f564--375-20
for i in $(seq -f "%010g" 375	394	); do rsync -a $input/25b2f1d5-7266-401c-bdfa-32944607f564/img${i}.jpg $out/25b2f1d5-7266-401c-bdfa-32944607f564--375-20; done

echo "55.50% - processing 25b2f1d5-7266-401c-bdfa-32944607f564--971-30"
mkdir -p $out/25b2f1d5-7266-401c-bdfa-32944607f564--971-30
for i in $(seq -f "%010g" 971	1000	); do rsync -a $input/25b2f1d5-7266-401c-bdfa-32944607f564/img${i}.jpg $out/25b2f1d5-7266-401c-bdfa-32944607f564--971-30; done

echo "56.00% - processing 26924e0d-c283-4fd1-b3c1-068ec88fa8a8--1837-40"
mkdir -p $out/26924e0d-c283-4fd1-b3c1-068ec88fa8a8--1837-40
for i in $(seq -f "%010g" 1837	1876	); do rsync -a $input/26924e0d-c283-4fd1-b3c1-068ec88fa8a8/img${i}.jpg $out/26924e0d-c283-4fd1-b3c1-068ec88fa8a8--1837-40; done

echo "56.50% - processing 26b4e1c1-54c7-419f-b2d3-63ea52fd3540--397-20"
mkdir -p $out/26b4e1c1-54c7-419f-b2d3-63ea52fd3540--397-20
for i in $(seq -f "%010g" 397	416	); do rsync -a $input/26b4e1c1-54c7-419f-b2d3-63ea52fd3540/img${i}.jpg $out/26b4e1c1-54c7-419f-b2d3-63ea52fd3540--397-20; done

echo "57.00% - processing 26ed1dd8-c21e-4c0a-95a5-6ff2999e4a85--131-10"
mkdir -p $out/26ed1dd8-c21e-4c0a-95a5-6ff2999e4a85--131-10
for i in $(seq -f "%010g" 131	140	); do rsync -a $input/26ed1dd8-c21e-4c0a-95a5-6ff2999e4a85/img${i}.jpg $out/26ed1dd8-c21e-4c0a-95a5-6ff2999e4a85--131-10; done

echo "57.50% - processing 26ed1dd8-c21e-4c0a-95a5-6ff2999e4a85--185-20"
mkdir -p $out/26ed1dd8-c21e-4c0a-95a5-6ff2999e4a85--185-20
for i in $(seq -f "%010g" 185	204	); do rsync -a $input/26ed1dd8-c21e-4c0a-95a5-6ff2999e4a85/img${i}.jpg $out/26ed1dd8-c21e-4c0a-95a5-6ff2999e4a85--185-20; done

echo "58.00% - processing 27a26e4b-4d6e-40e1-8dc2-7fecc14027a3--1853-30"
mkdir -p $out/27a26e4b-4d6e-40e1-8dc2-7fecc14027a3--1853-30
for i in $(seq -f "%010g" 1853	1882	); do rsync -a $input/27a26e4b-4d6e-40e1-8dc2-7fecc14027a3/img${i}.jpg $out/27a26e4b-4d6e-40e1-8dc2-7fecc14027a3--1853-30; done

echo "58.50% - processing 281915aa-bb7f-4b0d-8546-9526cc94b35c--1203-30"
mkdir -p $out/281915aa-bb7f-4b0d-8546-9526cc94b35c--1203-30
for i in $(seq -f "%010g" 1203	1232	); do rsync -a $input/281915aa-bb7f-4b0d-8546-9526cc94b35c/img${i}.jpg $out/281915aa-bb7f-4b0d-8546-9526cc94b35c--1203-30; done

echo "59.00% - processing 281915aa-bb7f-4b0d-8546-9526cc94b35c--159-10"
mkdir -p $out/281915aa-bb7f-4b0d-8546-9526cc94b35c--159-10
for i in $(seq -f "%010g" 159	168	); do rsync -a $input/281915aa-bb7f-4b0d-8546-9526cc94b35c/img${i}.jpg $out/281915aa-bb7f-4b0d-8546-9526cc94b35c--159-10; done

echo "59.50% - processing 28320e49-c141-41aa-87f4-4624cde39ec2--1521-40"
mkdir -p $out/28320e49-c141-41aa-87f4-4624cde39ec2--1521-40
for i in $(seq -f "%010g" 1521	1560	); do rsync -a $input/28320e49-c141-41aa-87f4-4624cde39ec2/img${i}.jpg $out/28320e49-c141-41aa-87f4-4624cde39ec2--1521-40; done

echo "60.00% - processing 2897e989-e955-4280-a230-4f8d288a3a4b--1004-40"
mkdir -p $out/2897e989-e955-4280-a230-4f8d288a3a4b--1004-40
for i in $(seq -f "%010g" 1004	1043	); do rsync -a $input/2897e989-e955-4280-a230-4f8d288a3a4b/img${i}.jpg $out/2897e989-e955-4280-a230-4f8d288a3a4b--1004-40; done

echo "60.50% - processing 28d6408f-899d-44d5-95f4-53ece4b2c98d--1115-10"
mkdir -p $out/28d6408f-899d-44d5-95f4-53ece4b2c98d--1115-10
for i in $(seq -f "%010g" 1115	1124	); do rsync -a $input/28d6408f-899d-44d5-95f4-53ece4b2c98d/img${i}.jpg $out/28d6408f-899d-44d5-95f4-53ece4b2c98d--1115-10; done

echo "61.00% - processing 28d6408f-899d-44d5-95f4-53ece4b2c98d--827-30"
mkdir -p $out/28d6408f-899d-44d5-95f4-53ece4b2c98d--827-30
for i in $(seq -f "%010g" 827	856	); do rsync -a $input/28d6408f-899d-44d5-95f4-53ece4b2c98d/img${i}.jpg $out/28d6408f-899d-44d5-95f4-53ece4b2c98d--827-30; done

echo "61.50% - processing 2911c0de-45ad-488a-a8e7-a445d37568ad--2475-40"
mkdir -p $out/2911c0de-45ad-488a-a8e7-a445d37568ad--2475-40
for i in $(seq -f "%010g" 2475	2514	); do rsync -a $input/2911c0de-45ad-488a-a8e7-a445d37568ad/img${i}.jpg $out/2911c0de-45ad-488a-a8e7-a445d37568ad--2475-40; done

echo "62.00% - processing 292b2447-407d-45b5-8663-8d92a3f03064--123-20"
mkdir -p $out/292b2447-407d-45b5-8663-8d92a3f03064--123-20
for i in $(seq -f "%010g" 123	142	); do rsync -a $input/292b2447-407d-45b5-8663-8d92a3f03064/img${i}.jpg $out/292b2447-407d-45b5-8663-8d92a3f03064--123-20; done

echo "62.50% - processing 292b2447-407d-45b5-8663-8d92a3f03064--257-30"
mkdir -p $out/292b2447-407d-45b5-8663-8d92a3f03064--257-30
for i in $(seq -f "%010g" 257	286	); do rsync -a $input/292b2447-407d-45b5-8663-8d92a3f03064/img${i}.jpg $out/292b2447-407d-45b5-8663-8d92a3f03064--257-30; done

echo "63.00% - processing 29381989-7537-4f7b-a2ea-612dde89bf68--18-40"
mkdir -p $out/29381989-7537-4f7b-a2ea-612dde89bf68--18-40
for i in $(seq -f "%010g" 18	57	); do rsync -a $input/29381989-7537-4f7b-a2ea-612dde89bf68/img${i}.jpg $out/29381989-7537-4f7b-a2ea-612dde89bf68--18-40; done

echo "63.50% - processing 2b8a34f9-10ac-4961-86e1-b09de51a15cd--2871-20"
mkdir -p $out/2b8a34f9-10ac-4961-86e1-b09de51a15cd--2871-20
for i in $(seq -f "%010g" 2871	2890	); do rsync -a $input/2b8a34f9-10ac-4961-86e1-b09de51a15cd/img${i}.jpg $out/2b8a34f9-10ac-4961-86e1-b09de51a15cd--2871-20; done

echo "64.00% - processing 2b8a34f9-10ac-4961-86e1-b09de51a15cd--3321-30"
mkdir -p $out/2b8a34f9-10ac-4961-86e1-b09de51a15cd--3321-30
for i in $(seq -f "%010g" 3321	3350	); do rsync -a $input/2b8a34f9-10ac-4961-86e1-b09de51a15cd/img${i}.jpg $out/2b8a34f9-10ac-4961-86e1-b09de51a15cd--3321-30; done

echo "64.50% - processing 2c8d4cf5-2f9e-49b6-a9ea-76f25dffd9bf--87-40"
mkdir -p $out/2c8d4cf5-2f9e-49b6-a9ea-76f25dffd9bf--87-40
for i in $(seq -f "%010g" 87	126	); do rsync -a $input/2c8d4cf5-2f9e-49b6-a9ea-76f25dffd9bf/img${i}.jpg $out/2c8d4cf5-2f9e-49b6-a9ea-76f25dffd9bf--87-40; done

echo "65.00% - processing 2c8fe342-f08b-474f-ae35-843cf583a19a--1088-40"
mkdir -p $out/2c8fe342-f08b-474f-ae35-843cf583a19a--1088-40
for i in $(seq -f "%010g" 1088	1127	); do rsync -a $input/2c8fe342-f08b-474f-ae35-843cf583a19a/img${i}.jpg $out/2c8fe342-f08b-474f-ae35-843cf583a19a--1088-40; done

echo "65.50% - processing 2cb383c0-4c33-4eed-9279-91521ff19cbc--157-20"
mkdir -p $out/2cb383c0-4c33-4eed-9279-91521ff19cbc--157-20
for i in $(seq -f "%010g" 157	176	); do rsync -a $input/2cb383c0-4c33-4eed-9279-91521ff19cbc/img${i}.jpg $out/2cb383c0-4c33-4eed-9279-91521ff19cbc--157-20; done

echo "66.00% - processing 2cb383c0-4c33-4eed-9279-91521ff19cbc--253-10"
mkdir -p $out/2cb383c0-4c33-4eed-9279-91521ff19cbc--253-10
for i in $(seq -f "%010g" 253	262	); do rsync -a $input/2cb383c0-4c33-4eed-9279-91521ff19cbc/img${i}.jpg $out/2cb383c0-4c33-4eed-9279-91521ff19cbc--253-10; done

echo "66.50% - processing 2da81c47-219f-41f6-8b6f-bdb30991b1b1--1449-10"
mkdir -p $out/2da81c47-219f-41f6-8b6f-bdb30991b1b1--1449-10
for i in $(seq -f "%010g" 1449	1458	); do rsync -a $input/2da81c47-219f-41f6-8b6f-bdb30991b1b1/img${i}.jpg $out/2da81c47-219f-41f6-8b6f-bdb30991b1b1--1449-10; done

echo "67.00% - processing 2da81c47-219f-41f6-8b6f-bdb30991b1b1--2545-20"
mkdir -p $out/2da81c47-219f-41f6-8b6f-bdb30991b1b1--2545-20
for i in $(seq -f "%010g" 2545	2564	); do rsync -a $input/2da81c47-219f-41f6-8b6f-bdb30991b1b1/img${i}.jpg $out/2da81c47-219f-41f6-8b6f-bdb30991b1b1--2545-20; done

echo "67.50% - processing 2dd9ae3d-c14f-455e-8f35-aa9ae193e4de--811-30"
mkdir -p $out/2dd9ae3d-c14f-455e-8f35-aa9ae193e4de--811-30
for i in $(seq -f "%010g" 811	840	); do rsync -a $input/2dd9ae3d-c14f-455e-8f35-aa9ae193e4de/img${i}.jpg $out/2dd9ae3d-c14f-455e-8f35-aa9ae193e4de--811-30; done

echo "68.00% - processing 2dd9ae3d-c14f-455e-8f35-aa9ae193e4de--865-20"
mkdir -p $out/2dd9ae3d-c14f-455e-8f35-aa9ae193e4de--865-20
for i in $(seq -f "%010g" 865	884	); do rsync -a $input/2dd9ae3d-c14f-455e-8f35-aa9ae193e4de/img${i}.jpg $out/2dd9ae3d-c14f-455e-8f35-aa9ae193e4de--865-20; done

echo "68.50% - processing 2deae04b-c991-425b-b00e-5fd6c0e7f117--1143-20"
mkdir -p $out/2deae04b-c991-425b-b00e-5fd6c0e7f117--1143-20
for i in $(seq -f "%010g" 1143	1162	); do rsync -a $input/2deae04b-c991-425b-b00e-5fd6c0e7f117/img${i}.jpg $out/2deae04b-c991-425b-b00e-5fd6c0e7f117--1143-20; done

echo "69.00% - processing 2deae04b-c991-425b-b00e-5fd6c0e7f117--717-30"
mkdir -p $out/2deae04b-c991-425b-b00e-5fd6c0e7f117--717-30
for i in $(seq -f "%010g" 717	746	); do rsync -a $input/2deae04b-c991-425b-b00e-5fd6c0e7f117/img${i}.jpg $out/2deae04b-c991-425b-b00e-5fd6c0e7f117--717-30; done

echo "69.50% - processing 2e0c85c2-62a2-415e-b175-0a89d78635d2--111-40"
mkdir -p $out/2e0c85c2-62a2-415e-b175-0a89d78635d2--111-40
for i in $(seq -f "%010g" 111	150	); do rsync -a $input/2e0c85c2-62a2-415e-b175-0a89d78635d2/img${i}.jpg $out/2e0c85c2-62a2-415e-b175-0a89d78635d2--111-40; done

echo "70.00% - processing 2e509e19-90ec-40f0-904e-2f5d0057ed6a--2787-30"
mkdir -p $out/2e509e19-90ec-40f0-904e-2f5d0057ed6a--2787-30
for i in $(seq -f "%010g" 2787	2816	); do rsync -a $input/2e509e19-90ec-40f0-904e-2f5d0057ed6a/img${i}.jpg $out/2e509e19-90ec-40f0-904e-2f5d0057ed6a--2787-30; done

echo "70.50% - processing 2e509e19-90ec-40f0-904e-2f5d0057ed6a--997-20"
mkdir -p $out/2e509e19-90ec-40f0-904e-2f5d0057ed6a--997-20
for i in $(seq -f "%010g" 997	1016	); do rsync -a $input/2e509e19-90ec-40f0-904e-2f5d0057ed6a/img${i}.jpg $out/2e509e19-90ec-40f0-904e-2f5d0057ed6a--997-20; done

echo "71.00% - processing 2f23b607-f2e6-4f58-85d3-004c840bead2--3689-40"
mkdir -p $out/2f23b607-f2e6-4f58-85d3-004c840bead2--3689-40
for i in $(seq -f "%010g" 3689	3728	); do rsync -a $input/2f23b607-f2e6-4f58-85d3-004c840bead2/img${i}.jpg $out/2f23b607-f2e6-4f58-85d3-004c840bead2--3689-40; done

echo "71.50% - processing 30dc1906-ebca-42f5-ae15-b371d7dbf658--1-20"
mkdir -p $out/30dc1906-ebca-42f5-ae15-b371d7dbf658--1-20
for i in $(seq -f "%010g" 1	20	); do rsync -a $input/30dc1906-ebca-42f5-ae15-b371d7dbf658/img${i}.jpg $out/30dc1906-ebca-42f5-ae15-b371d7dbf658--1-20; done

echo "72.00% - processing 30dc1906-ebca-42f5-ae15-b371d7dbf658--227-10"
mkdir -p $out/30dc1906-ebca-42f5-ae15-b371d7dbf658--227-10
for i in $(seq -f "%010g" 227	236	); do rsync -a $input/30dc1906-ebca-42f5-ae15-b371d7dbf658/img${i}.jpg $out/30dc1906-ebca-42f5-ae15-b371d7dbf658--227-10; done

echo "72.50% - processing 31e0707c-c7ab-4d75-bfb6-1da926ac0985--905-30"
mkdir -p $out/31e0707c-c7ab-4d75-bfb6-1da926ac0985--905-30
for i in $(seq -f "%010g" 905	934	); do rsync -a $input/31e0707c-c7ab-4d75-bfb6-1da926ac0985/img${i}.jpg $out/31e0707c-c7ab-4d75-bfb6-1da926ac0985--905-30; done

echo "73.00% - processing 31e21213-c8e6-40f4-b498-98db071819ed--11-30"
mkdir -p $out/31e21213-c8e6-40f4-b498-98db071819ed--11-30
for i in $(seq -f "%010g" 11	40	); do rsync -a $input/31e21213-c8e6-40f4-b498-98db071819ed/img${i}.jpg $out/31e21213-c8e6-40f4-b498-98db071819ed--11-30; done

echo "73.50% - processing 31e21213-c8e6-40f4-b498-98db071819ed--2135-10"
mkdir -p $out/31e21213-c8e6-40f4-b498-98db071819ed--2135-10
for i in $(seq -f "%010g" 2135	2144	); do rsync -a $input/31e21213-c8e6-40f4-b498-98db071819ed/img${i}.jpg $out/31e21213-c8e6-40f4-b498-98db071819ed--2135-10; done

echo "74.00% - processing 31e21213-c8e6-40f4-b498-98db071819ed--499-20"
mkdir -p $out/31e21213-c8e6-40f4-b498-98db071819ed--499-20
for i in $(seq -f "%010g" 499	518	); do rsync -a $input/31e21213-c8e6-40f4-b498-98db071819ed/img${i}.jpg $out/31e21213-c8e6-40f4-b498-98db071819ed--499-20; done

echo "74.50% - processing 31f09a1d-41ea-4b68-a25c-62e876cc9a3f--343-30"
mkdir -p $out/31f09a1d-41ea-4b68-a25c-62e876cc9a3f--343-30
for i in $(seq -f "%010g" 343	372	); do rsync -a $input/31f09a1d-41ea-4b68-a25c-62e876cc9a3f/img${i}.jpg $out/31f09a1d-41ea-4b68-a25c-62e876cc9a3f--343-30; done

echo "75.00% - processing 31f09a1d-41ea-4b68-a25c-62e876cc9a3f--409-10"
mkdir -p $out/31f09a1d-41ea-4b68-a25c-62e876cc9a3f--409-10
for i in $(seq -f "%010g" 409	418	); do rsync -a $input/31f09a1d-41ea-4b68-a25c-62e876cc9a3f/img${i}.jpg $out/31f09a1d-41ea-4b68-a25c-62e876cc9a3f--409-10; done

echo "75.50% - processing 31f09a1d-41ea-4b68-a25c-62e876cc9a3f--591-20"
mkdir -p $out/31f09a1d-41ea-4b68-a25c-62e876cc9a3f--591-20
for i in $(seq -f "%010g" 591	610	); do rsync -a $input/31f09a1d-41ea-4b68-a25c-62e876cc9a3f/img${i}.jpg $out/31f09a1d-41ea-4b68-a25c-62e876cc9a3f--591-20; done

echo "76.00% - processing 32104067-0ed4-4fdb-bda9-b3fb84889b32--1229-30"
mkdir -p $out/32104067-0ed4-4fdb-bda9-b3fb84889b32--1229-30
for i in $(seq -f "%010g" 1229	1258	); do rsync -a $input/32104067-0ed4-4fdb-bda9-b3fb84889b32/img${i}.jpg $out/32104067-0ed4-4fdb-bda9-b3fb84889b32--1229-30; done

echo "76.50% - processing 3288cd3f-316c-4b32-8236-c090d0aad48a--1689-40"
mkdir -p $out/3288cd3f-316c-4b32-8236-c090d0aad48a--1689-40
for i in $(seq -f "%010g" 1689	1728	); do rsync -a $input/3288cd3f-316c-4b32-8236-c090d0aad48a/img${i}.jpg $out/3288cd3f-316c-4b32-8236-c090d0aad48a--1689-40; done

echo "77.00% - processing 32b4bde9-7d56-4e58-a3e0-b97abc05303d--1131-20"
mkdir -p $out/32b4bde9-7d56-4e58-a3e0-b97abc05303d--1131-20
for i in $(seq -f "%010g" 1131	1150	); do rsync -a $input/32b4bde9-7d56-4e58-a3e0-b97abc05303d/img${i}.jpg $out/32b4bde9-7d56-4e58-a3e0-b97abc05303d--1131-20; done

echo "77.50% - processing 32b4bde9-7d56-4e58-a3e0-b97abc05303d--1931-30"
mkdir -p $out/32b4bde9-7d56-4e58-a3e0-b97abc05303d--1931-30
for i in $(seq -f "%010g" 1931	1960	); do rsync -a $input/32b4bde9-7d56-4e58-a3e0-b97abc05303d/img${i}.jpg $out/32b4bde9-7d56-4e58-a3e0-b97abc05303d--1931-30; done

echo "78.00% - processing 33957db3-d49d-4e0b-91e9-e917e731c563--816-40"
mkdir -p $out/33957db3-d49d-4e0b-91e9-e917e731c563--816-40
for i in $(seq -f "%010g" 816	855	); do rsync -a $input/33957db3-d49d-4e0b-91e9-e917e731c563/img${i}.jpg $out/33957db3-d49d-4e0b-91e9-e917e731c563--816-40; done

echo "78.50% - processing 344d41b9-70d9-4fe9-8a14-91ed84956c04--217-10"
mkdir -p $out/344d41b9-70d9-4fe9-8a14-91ed84956c04--217-10
for i in $(seq -f "%010g" 217	226	); do rsync -a $input/344d41b9-70d9-4fe9-8a14-91ed84956c04/img${i}.jpg $out/344d41b9-70d9-4fe9-8a14-91ed84956c04--217-10; done

echo "79.00% - processing 34b5b205-f969-4196-b036-8b3adf75d71b--1191-20"
mkdir -p $out/34b5b205-f969-4196-b036-8b3adf75d71b--1191-20
for i in $(seq -f "%010g" 1191	1210	); do rsync -a $input/34b5b205-f969-4196-b036-8b3adf75d71b/img${i}.jpg $out/34b5b205-f969-4196-b036-8b3adf75d71b--1191-20; done

echo "79.50% - processing 34b5b205-f969-4196-b036-8b3adf75d71b--1805-30"
mkdir -p $out/34b5b205-f969-4196-b036-8b3adf75d71b--1805-30
for i in $(seq -f "%010g" 1805	1834	); do rsync -a $input/34b5b205-f969-4196-b036-8b3adf75d71b/img${i}.jpg $out/34b5b205-f969-4196-b036-8b3adf75d71b--1805-30; done

echo "80.00% - processing 374832bf-f977-4e8b-b0e0-2f2ea1e38b5d--2095-20"
mkdir -p $out/374832bf-f977-4e8b-b0e0-2f2ea1e38b5d--2095-20
for i in $(seq -f "%010g" 2095	2114	); do rsync -a $input/374832bf-f977-4e8b-b0e0-2f2ea1e38b5d/img${i}.jpg $out/374832bf-f977-4e8b-b0e0-2f2ea1e38b5d--2095-20; done

echo "80.50% - processing 374832bf-f977-4e8b-b0e0-2f2ea1e38b5d--2141-30"
mkdir -p $out/374832bf-f977-4e8b-b0e0-2f2ea1e38b5d--2141-30
for i in $(seq -f "%010g" 2141	2170	); do rsync -a $input/374832bf-f977-4e8b-b0e0-2f2ea1e38b5d/img${i}.jpg $out/374832bf-f977-4e8b-b0e0-2f2ea1e38b5d--2141-30; done

echo "81.00% - processing 374832bf-f977-4e8b-b0e0-2f2ea1e38b5d--539-10"
mkdir -p $out/374832bf-f977-4e8b-b0e0-2f2ea1e38b5d--539-10
for i in $(seq -f "%010g" 539	548	); do rsync -a $input/374832bf-f977-4e8b-b0e0-2f2ea1e38b5d/img${i}.jpg $out/374832bf-f977-4e8b-b0e0-2f2ea1e38b5d--539-10; done

echo "81.50% - processing 38635888-7149-403b-8601-e3862b1ee8dc--2239-10"
mkdir -p $out/38635888-7149-403b-8601-e3862b1ee8dc--2239-10
for i in $(seq -f "%010g" 2239	2248	); do rsync -a $input/38635888-7149-403b-8601-e3862b1ee8dc/img${i}.jpg $out/38635888-7149-403b-8601-e3862b1ee8dc--2239-10; done

echo "82.00% - processing 38635888-7149-403b-8601-e3862b1ee8dc--2659-20"
mkdir -p $out/38635888-7149-403b-8601-e3862b1ee8dc--2659-20
for i in $(seq -f "%010g" 2659	2678	); do rsync -a $input/38635888-7149-403b-8601-e3862b1ee8dc/img${i}.jpg $out/38635888-7149-403b-8601-e3862b1ee8dc--2659-20; done

echo "82.50% - processing 38a28dd4-d0ec-48ea-a6d1-df539364f6ad--229-30"
mkdir -p $out/38a28dd4-d0ec-48ea-a6d1-df539364f6ad--229-30
for i in $(seq -f "%010g" 229	258	); do rsync -a $input/38a28dd4-d0ec-48ea-a6d1-df539364f6ad/img${i}.jpg $out/38a28dd4-d0ec-48ea-a6d1-df539364f6ad--229-30; done

echo "83.00% - processing 3a932fda-64a5-43f5-a531-b6614c667d53--49-10"
mkdir -p $out/3a932fda-64a5-43f5-a531-b6614c667d53--49-10
for i in $(seq -f "%010g" 49	58	); do rsync -a $input/3a932fda-64a5-43f5-a531-b6614c667d53/img${i}.jpg $out/3a932fda-64a5-43f5-a531-b6614c667d53--49-10; done

echo "83.50% - processing 3c2172aa-da7a-4ed2-859a-6e151a74a47f--171-10"
mkdir -p $out/3c2172aa-da7a-4ed2-859a-6e151a74a47f--171-10
for i in $(seq -f "%010g" 171	180	); do rsync -a $input/3c2172aa-da7a-4ed2-859a-6e151a74a47f/img${i}.jpg $out/3c2172aa-da7a-4ed2-859a-6e151a74a47f--171-10; done

echo "84.00% - processing 3c2172aa-da7a-4ed2-859a-6e151a74a47f--181-20"
mkdir -p $out/3c2172aa-da7a-4ed2-859a-6e151a74a47f--181-20
for i in $(seq -f "%010g" 181	200	); do rsync -a $input/3c2172aa-da7a-4ed2-859a-6e151a74a47f/img${i}.jpg $out/3c2172aa-da7a-4ed2-859a-6e151a74a47f--181-20; done

echo "84.50% - processing 3c2172aa-da7a-4ed2-859a-6e151a74a47f--915-30"
mkdir -p $out/3c2172aa-da7a-4ed2-859a-6e151a74a47f--915-30
for i in $(seq -f "%010g" 915	944	); do rsync -a $input/3c2172aa-da7a-4ed2-859a-6e151a74a47f/img${i}.jpg $out/3c2172aa-da7a-4ed2-859a-6e151a74a47f--915-30; done

echo "85.00% - processing 3cf51686-2f14-4f60-ad68-d5766292b02a--1265-30"
mkdir -p $out/3cf51686-2f14-4f60-ad68-d5766292b02a--1265-30
for i in $(seq -f "%010g" 1265	1294	); do rsync -a $input/3cf51686-2f14-4f60-ad68-d5766292b02a/img${i}.jpg $out/3cf51686-2f14-4f60-ad68-d5766292b02a--1265-30; done

echo "85.50% - processing 3cf51686-2f14-4f60-ad68-d5766292b02a--315-20"
mkdir -p $out/3cf51686-2f14-4f60-ad68-d5766292b02a--315-20
for i in $(seq -f "%010g" 315	334	); do rsync -a $input/3cf51686-2f14-4f60-ad68-d5766292b02a/img${i}.jpg $out/3cf51686-2f14-4f60-ad68-d5766292b02a--315-20; done

echo "86.00% - processing 3d01f5cf-ed49-4e06-bdb1-efcfed397e4c--1007-10"
mkdir -p $out/3d01f5cf-ed49-4e06-bdb1-efcfed397e4c--1007-10
for i in $(seq -f "%010g" 1007	1016	); do rsync -a $input/3d01f5cf-ed49-4e06-bdb1-efcfed397e4c/img${i}.jpg $out/3d01f5cf-ed49-4e06-bdb1-efcfed397e4c--1007-10; done

echo "86.50% - processing 3d01f5cf-ed49-4e06-bdb1-efcfed397e4c--613-30"
mkdir -p $out/3d01f5cf-ed49-4e06-bdb1-efcfed397e4c--613-30
for i in $(seq -f "%010g" 613	642	); do rsync -a $input/3d01f5cf-ed49-4e06-bdb1-efcfed397e4c/img${i}.jpg $out/3d01f5cf-ed49-4e06-bdb1-efcfed397e4c--613-30; done

echo "87.00% - processing 3d01f5cf-ed49-4e06-bdb1-efcfed397e4c--775-20"
mkdir -p $out/3d01f5cf-ed49-4e06-bdb1-efcfed397e4c--775-20
for i in $(seq -f "%010g" 775	794	); do rsync -a $input/3d01f5cf-ed49-4e06-bdb1-efcfed397e4c/img${i}.jpg $out/3d01f5cf-ed49-4e06-bdb1-efcfed397e4c--775-20; done

echo "87.50% - processing 3de33b96-42ce-410b-ad17-231c9da3d4f9--197-30"
mkdir -p $out/3de33b96-42ce-410b-ad17-231c9da3d4f9--197-30
for i in $(seq -f "%010g" 197	226	); do rsync -a $input/3de33b96-42ce-410b-ad17-231c9da3d4f9/img${i}.jpg $out/3de33b96-42ce-410b-ad17-231c9da3d4f9--197-30; done

echo "88.00% - processing 3de33b96-42ce-410b-ad17-231c9da3d4f9--565-20"
mkdir -p $out/3de33b96-42ce-410b-ad17-231c9da3d4f9--565-20
for i in $(seq -f "%010g" 565	584	); do rsync -a $input/3de33b96-42ce-410b-ad17-231c9da3d4f9/img${i}.jpg $out/3de33b96-42ce-410b-ad17-231c9da3d4f9--565-20; done

echo "88.50% - processing 3de33b96-42ce-410b-ad17-231c9da3d4f9--985-10"
mkdir -p $out/3de33b96-42ce-410b-ad17-231c9da3d4f9--985-10
for i in $(seq -f "%010g" 985	994	); do rsync -a $input/3de33b96-42ce-410b-ad17-231c9da3d4f9/img${i}.jpg $out/3de33b96-42ce-410b-ad17-231c9da3d4f9--985-10; done

echo "89.00% - processing 3e0c1106-4a84-4122-bab8-614e98674d31--1523-30"
mkdir -p $out/3e0c1106-4a84-4122-bab8-614e98674d31--1523-30
for i in $(seq -f "%010g" 1523	1552	); do rsync -a $input/3e0c1106-4a84-4122-bab8-614e98674d31/img${i}.jpg $out/3e0c1106-4a84-4122-bab8-614e98674d31--1523-30; done

echo "89.50% - processing 3eb0829f-6f33-46b6-b491-c6c53200de4b--1107-30"
mkdir -p $out/3eb0829f-6f33-46b6-b491-c6c53200de4b--1107-30
for i in $(seq -f "%010g" 1107	1136	); do rsync -a $input/3eb0829f-6f33-46b6-b491-c6c53200de4b/img${i}.jpg $out/3eb0829f-6f33-46b6-b491-c6c53200de4b--1107-30; done

echo "90.00% - processing 40d6688c-fc38-4b92-ab7a-62ebfb4d310e--981-30"
mkdir -p $out/40d6688c-fc38-4b92-ab7a-62ebfb4d310e--981-30
for i in $(seq -f "%010g" 981	1010	); do rsync -a $input/40d6688c-fc38-4b92-ab7a-62ebfb4d310e/img${i}.jpg $out/40d6688c-fc38-4b92-ab7a-62ebfb4d310e--981-30; done

echo "90.50% - processing 41ef17e8-5bfc-4caf-b883-fcb1b71cc106--1839-30"
mkdir -p $out/41ef17e8-5bfc-4caf-b883-fcb1b71cc106--1839-30
for i in $(seq -f "%010g" 1839	1868	); do rsync -a $input/41ef17e8-5bfc-4caf-b883-fcb1b71cc106/img${i}.jpg $out/41ef17e8-5bfc-4caf-b883-fcb1b71cc106--1839-30; done

echo "91.00% - processing 424b0a4d-8432-4eb3-ad43-334cdc7636a4--5349-30"
mkdir -p $out/424b0a4d-8432-4eb3-ad43-334cdc7636a4--5349-30
for i in $(seq -f "%010g" 5349	5378	); do rsync -a $input/424b0a4d-8432-4eb3-ad43-334cdc7636a4/img${i}.jpg $out/424b0a4d-8432-4eb3-ad43-334cdc7636a4--5349-30; done

echo "91.50% - processing 427e5454-59ad-41fe-88b3-9d77252132c4--1025-30"
mkdir -p $out/427e5454-59ad-41fe-88b3-9d77252132c4--1025-30
for i in $(seq -f "%010g" 1025	1054	); do rsync -a $input/427e5454-59ad-41fe-88b3-9d77252132c4/img${i}.jpg $out/427e5454-59ad-41fe-88b3-9d77252132c4--1025-30; done

echo "92.00% - processing 47249b6c-5d9c-44dd-9ab9-a68fa59ccb3f--5967-30"
mkdir -p $out/47249b6c-5d9c-44dd-9ab9-a68fa59ccb3f--5967-30
for i in $(seq -f "%010g" 5967	5996	); do rsync -a $input/47249b6c-5d9c-44dd-9ab9-a68fa59ccb3f/img${i}.jpg $out/47249b6c-5d9c-44dd-9ab9-a68fa59ccb3f--5967-30; done

echo "92.50% - processing 495172da-f1f8-459e-bffa-e8cec7dfc52e--1957-30"
mkdir -p $out/495172da-f1f8-459e-bffa-e8cec7dfc52e--1957-30
for i in $(seq -f "%010g" 1957	1986	); do rsync -a $input/495172da-f1f8-459e-bffa-e8cec7dfc52e/img${i}.jpg $out/495172da-f1f8-459e-bffa-e8cec7dfc52e--1957-30; done

echo "93.00% - processing 4afc7a4d-9874-43ad-84cb-7209530292b9--2393-30"
mkdir -p $out/4afc7a4d-9874-43ad-84cb-7209530292b9--2393-30
for i in $(seq -f "%010g" 2393	2422	); do rsync -a $input/4afc7a4d-9874-43ad-84cb-7209530292b9/img${i}.jpg $out/4afc7a4d-9874-43ad-84cb-7209530292b9--2393-30; done

echo "93.50% - processing 4c270344-1a5c-4a86-bf76-e7d98c8c5ebb--513-30"
mkdir -p $out/4c270344-1a5c-4a86-bf76-e7d98c8c5ebb--513-30
for i in $(seq -f "%010g" 513	542	); do rsync -a $input/4c270344-1a5c-4a86-bf76-e7d98c8c5ebb/img${i}.jpg $out/4c270344-1a5c-4a86-bf76-e7d98c8c5ebb--513-30; done

echo "94.00% - processing 4e4b4b97-33c7-4760-a5a2-92b60a6f6137--5135-30"
mkdir -p $out/4e4b4b97-33c7-4760-a5a2-92b60a6f6137--5135-30
for i in $(seq -f "%010g" 5135	5164	); do rsync -a $input/4e4b4b97-33c7-4760-a5a2-92b60a6f6137/img${i}.jpg $out/4e4b4b97-33c7-4760-a5a2-92b60a6f6137--5135-30; done

echo "94.50% - processing 4fd8696f-5629-4624-b080-7448d7cacedf--5237-30"
mkdir -p $out/4fd8696f-5629-4624-b080-7448d7cacedf--5237-30
for i in $(seq -f "%010g" 5237	5266	); do rsync -a $input/4fd8696f-5629-4624-b080-7448d7cacedf/img${i}.jpg $out/4fd8696f-5629-4624-b080-7448d7cacedf--5237-30; done

echo "95.00% - processing 50477443-1f79-4bce-9bcd-8e468f1b5779--3277-30"
mkdir -p $out/50477443-1f79-4bce-9bcd-8e468f1b5779--3277-30
for i in $(seq -f "%010g" 3277	3306	); do rsync -a $input/50477443-1f79-4bce-9bcd-8e468f1b5779/img${i}.jpg $out/50477443-1f79-4bce-9bcd-8e468f1b5779--3277-30; done

echo "95.50% - processing 56723ca2-d092-4a6d-aa2a-25669f6644f7--5223-30"
mkdir -p $out/56723ca2-d092-4a6d-aa2a-25669f6644f7--5223-30
for i in $(seq -f "%010g" 5223	5252	); do rsync -a $input/56723ca2-d092-4a6d-aa2a-25669f6644f7/img${i}.jpg $out/56723ca2-d092-4a6d-aa2a-25669f6644f7--5223-30; done

echo "96.00% - processing 578145af-2e9c-4841-a215-4d067ff5057d--325-30"
mkdir -p $out/578145af-2e9c-4841-a215-4d067ff5057d--325-30
for i in $(seq -f "%010g" 325	354	); do rsync -a $input/578145af-2e9c-4841-a215-4d067ff5057d/img${i}.jpg $out/578145af-2e9c-4841-a215-4d067ff5057d--325-30; done

echo "96.50% - processing 5a1abc35-64b1-4686-ba62-565d883e128b--2229-30"
mkdir -p $out/5a1abc35-64b1-4686-ba62-565d883e128b--2229-30
for i in $(seq -f "%010g" 2229	2258	); do rsync -a $input/5a1abc35-64b1-4686-ba62-565d883e128b/img${i}.jpg $out/5a1abc35-64b1-4686-ba62-565d883e128b--2229-30; done

echo "97.00% - processing 5c16c4b4-8065-410a-8160-8da3b2d0e57e--1209-30"
mkdir -p $out/5c16c4b4-8065-410a-8160-8da3b2d0e57e--1209-30
for i in $(seq -f "%010g" 1209	1238	); do rsync -a $input/5c16c4b4-8065-410a-8160-8da3b2d0e57e/img${i}.jpg $out/5c16c4b4-8065-410a-8160-8da3b2d0e57e--1209-30; done

echo "97.50% - processing 5ce1f996-f784-4c9e-8208-e370bf8adf39--3517-30"
mkdir -p $out/5ce1f996-f784-4c9e-8208-e370bf8adf39--3517-30
for i in $(seq -f "%010g" 3517	3546	); do rsync -a $input/5ce1f996-f784-4c9e-8208-e370bf8adf39/img${i}.jpg $out/5ce1f996-f784-4c9e-8208-e370bf8adf39--3517-30; done

echo "98.00% - processing 5e27061d-df4e-49c2-a85f-c2296666b191--2163-30"
mkdir -p $out/5e27061d-df4e-49c2-a85f-c2296666b191--2163-30
for i in $(seq -f "%010g" 2163	2192	); do rsync -a $input/5e27061d-df4e-49c2-a85f-c2296666b191/img${i}.jpg $out/5e27061d-df4e-49c2-a85f-c2296666b191--2163-30; done

echo "98.50% - processing 67759ea1-52a2-45ee-b129-f482532d83df--1007-30"
mkdir -p $out/67759ea1-52a2-45ee-b129-f482532d83df--1007-30
for i in $(seq -f "%010g" 1007	1036	); do rsync -a $input/67759ea1-52a2-45ee-b129-f482532d83df/img${i}.jpg $out/67759ea1-52a2-45ee-b129-f482532d83df--1007-30; done

echo "99.00% - processing 67f9b6ac-0137-4787-8dba-b30aad329898--947-30"
mkdir -p $out/67f9b6ac-0137-4787-8dba-b30aad329898--947-30
for i in $(seq -f "%010g" 947	976	); do rsync -a $input/67f9b6ac-0137-4787-8dba-b30aad329898/img${i}.jpg $out/67f9b6ac-0137-4787-8dba-b30aad329898--947-30; done

echo "99.50% - processing 68205f0f-9b30-4d81-986d-f8816e70bddd--561-30"
mkdir -p $out/68205f0f-9b30-4d81-986d-f8816e70bddd--561-30
for i in $(seq -f "%010g" 561	590	); do rsync -a $input/68205f0f-9b30-4d81-986d-f8816e70bddd/img${i}.jpg $out/68205f0f-9b30-4d81-986d-f8816e70bddd--561-30; done

echo "100.00% - Finish processing"