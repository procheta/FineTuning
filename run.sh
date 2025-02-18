for ita in 0.001 0.005 0.01 .05 .1 .5 1 5 10
do
     echo $ita >> output.txt
     python llama_dGClip.py 0.000001 $ita >> output.txt
done
