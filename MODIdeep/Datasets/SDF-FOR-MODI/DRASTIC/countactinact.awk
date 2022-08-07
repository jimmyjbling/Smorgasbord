BEGIN { for(i=5;i<=64;i++) {k1[i-4]=0; k0[i-4]=0;}}  

{ for(i=5;i<=64;i++) { if($i==1) k1[i-4]++; if($i==0) k0[i-4]++; }} 

END { for(i=5;i<=64;i++) print i-4,k1[i-4],k0[i-4],k1[i-4]+k0[i-4]; }

