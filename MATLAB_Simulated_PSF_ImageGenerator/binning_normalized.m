
function [I0_teo,I90_teo,I45_teo,I135_teo]=binning_normalized(Ix_plane,Iy_plane,Iu_plane,Iv_plane,binsize,X,Y)

%% resize with binsize
IxtempT=bin_image(Ix_plane,binsize);
IytempT=bin_image(Iy_plane,binsize);
IutempT=bin_image(Iu_plane,binsize);
IvtempT=bin_image(Iv_plane,binsize);
newX=X(1:binsize:end);
newY=Y(1:binsize:end);

Xlims=(newX>-1)&(newX<1);
Ylims=(newY>-1)&(newY<1);
factor_correction_0=sum(Ix_plane,"all")/sum(IxtempT(Ylims,Xlims),"all");
factor_correction_90=sum(Iy_plane,"all")/sum(IytempT(Ylims,Xlims),"all");
factor_correction_45=sum(Iu_plane,"all")/sum(IutempT(Ylims,Xlims),"all");
factor_correction_135=sum(Iv_plane,"all")/sum(IvtempT(Ylims,Xlims),"all");

I0_teo=IxtempT(Ylims,Xlims).*factor_correction_0;
I90_teo=IytempT(Ylims,Xlims).*factor_correction_90;
I45_teo=IutempT(Ylims,Xlims).*factor_correction_45;
I135_teo=IvtempT(Ylims,Xlims).*factor_correction_135;