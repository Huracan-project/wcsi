# Find all points with closed contours of MSLP
# (at least 1hPa within 5 degrees)

DetectNodes \
    --in_data_list "in_data_list_mslp.txt" \
    --out "mslp-nodes" \
    --searchbymin "mslp" \
    --closedcontourcmd "mslp,100,5,0" \
    --lonname "longitude" \
    --latname "latitude" \
    --outputcmd "mslp,max,0"

cat mslp-nodes*.dat > mslp-nodes.csv
