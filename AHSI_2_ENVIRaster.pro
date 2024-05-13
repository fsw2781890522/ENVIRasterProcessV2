pro AHSI_2_ENVIRaster

  e = ENVI()
  
  sav_fn = "D:\SOFTWARE\ENVI60\ENVI60\extensions\ChinaSatellitesSupport\envi_china_satellites_support.sav"
  
  root_dir = 'J:\GROUP2_AHSI_original'
  out_dir = 'J:\GROUP2_AHSI_img'
  
  if file_test(out_dir) ne 1 then file_mkdir, out_dir
  
  fn_list = get_fn_list(root_dir)
  
  forEach fn, fn_list do begin
    
    ext = get_ext(fn)
    basename = file_basename(fn, ext)
;    base_str = strsplit(basename, '_', /extract)
;    basename = strjoin(base_str[0:4], '_')
    
    raster = read_raster(fn, sav_fn)
    data = raster.getData()
    meta = raster.metadata
    geo = raster.spatialRef
    interleave = raster.interleave
    
    o_fn = strcompress(out_dir + '\' + basename + '.img')
    
    if file_test(o_fn) then continue
    
    new_raster = ENVIRaster(data, URI=o_fn, spatialRef=geo, metadata=meta, interleave=interleave)
    new_raster.save
    
  endforeach

end

function get_ext, fn

  return, '.' + (strsplit(fn, '.', /extract))[-1]

end

function get_fn_list, root_dir

  fn_list = []
  files = File_Search(root_dir + '/*', '*.{xml,dat,img}', count=count)

  for i = 0, count-1 do begin
    file_path = files[i]
    if (StrPos(files[i], 'Check')) eq -1 and (StrPos(files[i], 'describe')) eq -1 then begin
      fn_list = [fn_list, files[i]]
    endif
  endfor

  if n_elements(fn_list) gt 0 then begin
    return, fn_list
  endif else begin
    print, 'No standard ENVIRaster[*.dat/*.img] or [*.xml] file of China Satellites found.'
    return, 0
  endelse

end

function read_raster, fn, sav_fn

  e = ENVI()
  Restore, sav_fn
  ext = (strsplit(fn, '.', /extract))[-1]
  if ext eq 'dat' or ext eq 'img' then begin
    return, e.openRaster(fn)
  endif else if ext eq 'xml' then begin
    src = ENVIOpenChinaRaster(fn)
    vn = src[1]
    sw = src[2]
    vn.close
    sw.close
    raster = src[0]
    return, raster
  endif else begin
    print, 'Failed to open. Please present standard ENVIRaster[*.dat/*.img] or [*.xml] file of China Satellites'
    return, 0
  endelse

end