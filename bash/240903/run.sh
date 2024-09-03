#!/bin/bash

# 각 스크립트를 순차적으로 실행
bash bash/240903/decom_test_coord_depth_2.sh
bash bash/240903/decom_test_coord_depth_w256.sh
bash bash/240903/decom_test_batch.sh
bash bash/240903/decom_test_coord_depth_appendix.sh

echo "모든 스크립트가 완료되었습니다."
