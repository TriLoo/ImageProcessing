function fusef(action)
% GUI- Funktion fuer fusetool

% Colormap setzen
m = gray(256);

% was ist zu tun ?
switch(action)

	% Initialisierungen %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	case('create')
  	pathname = '';
    % Speichern in Userdata
    data.pathname = pathname;
    set(gcbf,'Userdata',data);
     	  
 	% Bild A laden %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 	case('loadA')
 	  data 			= get(gcbf,'Userdata');
 	  pathname 	= data.pathname;
   	[filename, pathname] = uigetfile([pathname '*.*'], 'Load input image A');
   	if filename~=0
    	[M1,ma] = imread([pathname, filename]);
    	if isind(M1) & ~isempty(ma)  
     		M1 = 256*double(ind2gray(M1,ma));
     	else
     	  if isgray(M1)
     	    M1 = double(M1);
     	  else
     		 	M1 = double(rgb2gray(M1));
     		end; 	
     	end;	
     	% Speichern in Userdata
     	data.M1 = M1;
     	data.pathname = pathname;
     	set(gcbf,'Userdata',data);    	
     	% Achsen setzen
     	set(gcbf,'CurrentAxes',findobj(gcbf,'Tag','Axes1'));
     	% Bild anzeigen
     	image(M1);
     	axis image;
     	% Colormap gray
     	colormap(m);
     	% Achsen neu setzen (Matlab-Bug)
     	set(gca, 'Tag', 'Axes1');
     	% Ticks loeschen
     	set(findobj(gcbf,'Tag','Axes1'),'Xticklabel',[],'Yticklabel',[]);
     	% evtl Grid anschalten
     	fusef('gridonoff')     
      % evtl Message ausgeben
     	fusef('messize');
  	end;
  
 	% Bild B laden %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 	case('loadB')
 	  data 			= get(gcbf,'Userdata');
 	  pathname 	= data.pathname;
   	[filename, pathname] = uigetfile([pathname '*.*'], 'Load input image B');
   	if filename~=0
     	[M2,ma] = imread([pathname, filename]);
    	if isind(M2) & ~isempty(ma)  
     		M2 = 256*double(ind2gray(M2,ma));
     	else
     	  if isgray(M2)
     	    M2 = double(M2);
     	  else
     		 	M2 = double(rgb2gray(M2));
     		end; 	
     	end;
     	% Speichern in Userdata
     	data.M2 = M2;
     	data.pathname = pathname;
     	set(gcbf,'Userdata',data);
     	% Achsen setzen
     	set(gcbf,'CurrentAxes',findobj(gcbf,'Tag','Axes2'));
     	% Bild anzeigen
     	image(M2);
     	axis image;
     	% Colormap gray
     	colormap(m);
     	% Achsen neu setzen (Matlab-Bug)
     	set(gca, 'Tag', 'Axes2');
     	% Ticks loeschen
     	set(findobj(gcbf,'Tag','Axes2'),'Xticklabel',[],'Yticklabel',[]);
     	% evtl Grid anschalten
     	fusef('gridonoff')
      % evtl Message ausgeben     
     	fusef('messize');
  end;
 
  % Bild speichern %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  case('saveF')
   	data 			= get(gcbf,'Userdata');
  	if (isfield(data,'F'))
 	  	pathname 	= data.pathname;
    	F 				= data.F;
    	if(~isempty(F))
     		[filename, pathname] = uiputfile([pathname 'fusion.bmp'], 'Save fused image');
     		if (filename(~0))
       		% Clippen des Bildes
       		F(find(F<0))		=	0;
       		F(find(F>255))	=	255;
       		imwrite(F+1,m,[pathname, filename],'bmp');
      	end;   
    	end; 
   	end;
   	  
  % Display anpassen an Fusionsauswahl %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  case('SelDisp')
    % Sichtbar/ unsichtbar AreaBox
    cf1 = get(findobj(gcbf,'Tag','CoeffMenu'),'value');
    cf2 = get(findobj(gcbf,'Tag','FusTypMenu'),'value');
    if(cf2<5)
      set(findobj(gcbf,'Tag','CoeffMenu'),'visible','off');
      set(findobj(gcbf,'Tag','AreaMenu'),'visible','off');
      set(findobj(gcbf,'Tag','SelBaseMenu'),'visible','off');
      set(findobj(gcbf,'Tag','DecompMenu'),'visible','off');
    else
      set(findobj(gcbf,'Tag','CoeffMenu'),'visible','on');
      set(findobj(gcbf,'Tag','SelBaseMenu'),'visible','on');
      set(findobj(gcbf,'Tag','DecompMenu'),'visible','on');  
      if(cf1==2 | cf1==3)
        set(findobj(gcbf,'Tag','AreaMenu'),'visible','on');
      else
        set(findobj(gcbf,'Tag','AreaMenu'),'visible','off');
      end;
    end;  
  
  % Fusion durchfuehren %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
  case('fusion')
  	data 	= get(gcbf,'Userdata');
 	  if (~isfield(data,'M1') | ~isfield(data,'M2'))
 	    fusef('messinput');
 	  else   
 	  	M1		= data.M1;
 	  	M2		= data.M2;
 	  	% Check inputs
    	[z1 s1] = size(M1);
    	[z2 s2] = size(M2);
    	if (z1~=z2 | s1~=s2)
      	fusef('messize');
    	else  
      	fusef('messtart')      	
      	% Zuerst Parameter auslesen  
      	aw = get(findobj(gcbf,'Tag','FusTypMenu'),'value');
      	mp = get(findobj(gcbf,'Tag','SelBaseMenu'),'value');
      	zt = get(findobj(gcbf,'Tag','DecompMenu'),'value');
      	cf = get(findobj(gcbf,'Tag','CoeffMenu'),'value');
      	ar = 1+2*get(findobj(gcbf,'Tag','AreaMenu'),'value');
      	cc = [cf ar];
      	set(gcbf,'pointer','watch');
      	% Auswahl Fusionsverfahren
      	switch(aw)
        	case 1,  F = selb(M1,M2,3);
        	case 2,  F = fuse_pca(M1,M2); 
        	case 3,  F = selc(M1,M2,4);
        	case 4,  F = -selc(-M1,-M2,4);
        	case 5,  F = fuse_lap(M1,M2,zt,cc,mp);
        	case 6,  F = fuse_fsd(M1,M2,zt,cc,mp);
        	case 7,  F = fuse_rat(M1,M2,zt,cc,mp);
        	case 8,  F = fuse_con(M1,M2,zt,cc,mp);
        	case 9,  F = fuse_gra(M1,M2,zt,cc,mp); 
        	case 10, F = fuse_dwb(M1,M2,zt,cc,mp);
        	case 11, F = fuse_sih(M1,M2,zt,cc,mp);      
        	case 12, F = fuse_mod(M1,M2,zt,cc,mp);            	
      	end;
      	set(gcbf,'pointer','arrow');
      	fusef('messtop');
      	% Speichern in Userdata
     		data.F = F;
     		set(gcbf,'Userdata',data);  
      	% Bild anzeigen
      	set(gcbf,'CurrentAxes',findobj(gcbf,'Tag','Axes3'));
      	image(F);
      	axis image;
      	% Achsen neu setzen
      	set(gca, 'Tag', 'Axes3');
      	colormap(m);
      	% Ticks loeschen
      	set(findobj(gcbf,'Tag','Axes3'),'Xticklabel',[],'Yticklabel',[]);
      	% evtl Neue Figure
      	fusef('neufig');
  	    % evtl Grid
    	  fusef('gridonoff')
    	end;
    end;	
    
  % Zoom %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  case('zoom')
    if(get(findobj(gcbf,'Tag','ZoomBox'),'value')==1)
      zoom on;
    else 
      zoom off;
    end;
    
  % Grid %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  case('gridonoff')
    if(get(findobj(gcbf,'Tag','GridBox'),'value')==1)
      set(findobj(gcbf,'Tag','Axes1'),'Xcolor','y','ycolor','y','xgrid','on','ygrid','on');
      set(findobj(gcbf,'Tag','Axes2'),'Xcolor','y','ycolor','y','xgrid','on','ygrid','on');
      set(findobj(gcbf,'Tag','Axes3'),'Xcolor','y','ycolor','y','xgrid','on','ygrid','on');
    else
     set(findobj(gcbf,'Tag','Axes1'),'Xcolor','k','ycolor','k','xgrid','off','ygrid','off');
     set(findobj(gcbf,'Tag','Axes2'),'Xcolor','k','ycolor','k','xgrid','off','ygrid','off');
     set(findobj(gcbf,'Tag','Axes3'),'Xcolor','k','ycolor','k','xgrid','off','ygrid','off');
    end;
    
  % Messages %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  case('messize') 
  	data 	= get(gcbf,'Userdata');
 	  if (~isfield(data,'M1') | ~isfield(data,'M2'))
 	    fusef('messinput');
 	  else
 	   	M1 = data.M1;
 	   	M2 = data.M2; 
     	if(~isempty(M1) & ~isempty(M2))
       	[z1 s1] = size(M1); 
       	[z2 s2] = size(M2); 
       	if (z1~=z2 | s1~=s2)
        	set(findobj(gcbf,'Tag','MessText'),'String','Input images are not of same size','ForegroundColor','r');
       	else
         	set(findobj(gcbf,'Tag','MessText'),'String','','ForegroundColor','y');
       	end;
     	end;
    end;
  case('messtart')
   	set(findobj(gcbf,'Tag','MessText'),'String','Computing. Wait please','ForegroundColor','y');
  case('messtop')
   	set(findobj(gcbf,'Tag','MessText'),'String','','ForegroundColor','y'); 
  case('messinput')
   	set(findobj(gcbf,'Tag','MessText'),'String','Not enough input images','ForegroundColor','r'); 
     
  % Neue Figure %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  case('neufig')
    nf = get(findobj(gcbf,'Tag','FigBox'),'value');   
    if (nf==1)
    	data 	= get(gcbf,'Userdata');
 	    F     = data.F;
      figure; colormap(m);
      image(F); axis image; axis off;
      aw = get(findobj(gcbf,'Tag','FusTypMenu'),'value');
      switch(aw)
        case 1,  	title('Fusion result: Addition');
        case 2,  	title('Fusion result: PCA Method');
        case 3,  	title('Fusion result: Maximum selection');
        case 4,  	title('Fusion result: Minimum selection');
        case 5,  	title('Fusion result: Laplacian Pyramid');
        case 6,  	title('Fusion result: FSD Pyramid');
        case 7,  	title('Fusion result: Ratio Pyramid');
        case 8,  	title('Fusion result: Contrast Pyramid');
        case 9,  	title('Fusion result: Gradient Pyramid');
        case 10,  title('Fusion result: DWT with DBSS(2,2) wavelet');
        case 11,  title('Fusion result: SIDWT with Haar wavelet');
        case 12,  title('Fusion result: Morphological Pyramid');
      end;  
    end;     
end;
    