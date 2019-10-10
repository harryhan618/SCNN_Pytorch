root=/home/lion/SCNN_Pytorch/
exp=$1
data_dir=/home/lion/Dataset/CULane/data/CULane/
detect_dir=${root}/experiments/${exp}/coord_output/
bin_dir=${root}/utils/lane_evaluation/CULane

w_lane=30;
iou=0.5;  # Set iou to 0.3 or 0.5
im_w=1640
im_h=590
frame=1
list0=${data_dir}list/test_split/test0_normal.txt
list1=${data_dir}list/test_split/test1_crowd.txt
list2=${data_dir}list/test_split/test2_hlight.txt
list3=${data_dir}list/test_split/test3_shadow.txt
list4=${data_dir}list/test_split/test4_noline.txt
list5=${data_dir}list/test_split/test5_arrow.txt
list6=${data_dir}list/test_split/test6_curve.txt
list7=${data_dir}list/test_split/test7_cross.txt
list8=${data_dir}list/test_split/test8_night.txt
out0=${detect_dir}../evaluate/out0_normal.txt
out1=${detect_dir}../evaluate/out1_crowd.txt
out2=${detect_dir}../evaluate/out2_hlight.txt
out3=${detect_dir}../evaluate/out3_shadow.txt
out4=${detect_dir}../evaluate/out4_noline.txt
out5=${detect_dir}../evaluate/out5_arrow.txt
out6=${detect_dir}../evaluate/out6_curve.txt
out7=${detect_dir}../evaluate/out7_cross.txt
out8=${detect_dir}../evaluate/out8_night.txt
${bin_dir}/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list0 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out0
${bin_dir}/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list1 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out1
${bin_dir}/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list2 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out2
${bin_dir}/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list3 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out3
${bin_dir}/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list4 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out4
${bin_dir}/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list5 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out5
${bin_dir}/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list6 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out6
${bin_dir}/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list7 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out7
${bin_dir}/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list8 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out8
cat ${detect_dir}/../evaluate/out*.txt > ${detect_dir}/../evaluate/${exp}_iou${iou}_split.txt
