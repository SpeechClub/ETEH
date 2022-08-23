wavdir=dev/WAV
txtdir=dev/TXT

segwavdir=dev/segWAV
reffile=dev/ref.gb.txt

mkdir -p $segwavdir
rm $reffile
for fname in `ls $txtdir | sed "s/\.txt//"` ; do
	awk -F '\t' -v FNAME=$fname '$2!=0{print FNAME "\t" $1 "\t" $4}' $txtdir/$fname.txt | sed -r "s/\[([^,]*),([^]]*)\]/\1\t\2/" | sed "s/《/ /g" |sed "s/》/ /g"|sed "s/…/ /g"|sed "s/~/ /g"| sed "s/，/ /g"  | sed "s/。/ /g" | sed "s/·/ /g"  | sed "s/？/ /g" | sed "s/！/ /g" | sed "s/；/ /g" | sed "s/：/ /g" | sed "s/“/ /g" | sed "s/”/ /g"  | sed "s/‘/'/g" | sed "s/’/'/g"| sed "s/\`/'/g" | sed "s/,/ /g"  | sed "s/?/ /g" | sed "s/\!/ /g" | sed "s/;/ /g" | sed "s/\"/ /g"| sed "s/:/ /g" | sed "s/　/ /g" | sed "s/ / /g" | sed "s/\r//g" > all_segs
	awk -F '\t' -v FNAME=$fname -v WAV=$wavdir/$fname.wav -v SEGWAVDIR=$segwavdir '{print "sox " WAV " " SEGWAVDIR "/" FNAME "-" $2 "-" $3 ".wav trim " $2 " " $3-$2}' all_segs | sh
	awk -F '\t' -v FNAME=$fname '{print $4 " (" FNAME "-" $2 "-" $3 ".wav)"}' all_segs | sed "s/\[[^]]*\]//g" | iconv -f utf8 -t cp936 >> $reffile
done
