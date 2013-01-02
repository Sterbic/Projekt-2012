echo -e "###\nWarning: ETA for this test is cca 15h!\n###\n"
./SWalign Sequences/6/BA000046.3.fna Sequences/6/NC_000021.7.fna 1 -3 -5 -2
echo -e "###\nAlignment score should be: 88353 at [1072950, 722725]\n###\n"
