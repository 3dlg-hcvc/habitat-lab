echo "starting setup ${SLURM_NODEID}"

rm -rf /scratch/hanxiao/scene-builder-datasets/fphab/habitat-lab/data

cp /home/hanxiao/data.zip $SLURM_TMPDIR/
unzip $SLURM_TMPDIR/data.zip -d $SLURM_TMPDIR/

ln -s $SLURM_TMPDIR/data /scratch/hanxiao/scene-builder-datasets/fphab/habitat-lab/data

touch $SLURM_TMPDIR/$SLURM_NODEID.txt

echo $SLURM_NODEID
echo "done"
