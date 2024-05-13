pro ASTER_2_ENVIRaster

  e = ENVI()
  
  root_dir = 'H:\20240411_Group2_ASTER\ASTER_original_HDF'
  temp_dir = 'H:\20240411_Group2_ASTER\ASTER_original_temp_img'
  out_dir = 'H:\20240411_Group2_ASTER\ASTER_original_img'
  
  csv_fn = "H:\20240411_Group2_ASTER\filename.csv"
  
;  csv_fn = ''
 
  forEach dir, [temp_dir, out_dir] do begin
    if file_test(dir) ne 1 then file_mkdir, dir
  endforeach
  
  
  fn_list = get_fn_list(root_dir, csv_fn)
  
  false_list = []
  time_list = []
  file_list = []
  no_exist_list = []
  
  for i = 0, n_elements(fn_list)-1 do begin
    
    fn = fn_list[i]
    
    if file_test(fn) then begin
        
        start_time = systime(/seconds)
        
        basename = file_basename(fn, '.hdf')
        
        temp_fn = strcompress(temp_dir + '\' + basename + '.img', /remove_all)
        stack_fn = strcompress(out_dir + '\' + basename + '.img', /remove_all)

        if file_test(stack_fn) then continue
        
        img = e.OpenRaster(fn)
        help, img
  
        if n_elements(img) lt 3 then begin
  
          print, 'False Data!'
          false_list = [false_list, [basename]]
  
        endif else begin
  
          VNIR = img[0]
          SWIR = img[1]
          TIR = img[2]
  
          meta1 = VNIR.metadata
          meta2 = SWIR.metadata
          meta3 = TIR.metadata
          help, meta1
          tags = meta1.tags

          geo = VNIR.spatialRef
          gain1 = meta1['DATA GAIN VALUES']
          offset1 = meta1['DATA OFFSET VALUES']
          gain2 = meta2['DATA GAIN VALUES']
          offset2 = meta2['DATA OFFSET VALUES']
  
          time = meta1['ACQUISITION TIME']
          time = ENVITime(ACQUISITION = time)
          help, time
  
  
          if n_elements(gain1) eq 3 then begin
            if n_elements(gain2) ne 6 then begin
  
              print, 'False Data!'
              false_list = [false_list, [basename]]
  
            endif else begin
  
              time_list = [time_list, [time]]
              file_list = [file_list, [basename]]
     
              temp = Layer_Stack([VNIR, SWIR, TIR], temp_fn)
              help, temp
              
              new_meta = ENVIRasterMetadata()
              
              forEach tag, tags do begin
                if n_elements(meta1[tag]) eq 1 then begin
                  value = meta1[tag]
                  help, tag, meta1[tag]
                  new_meta.addItem, tag, value
                endif else begin
                  if meta2.hasTag(tag) and meta3.hasTag(tag) then begin
                    value1 = meta1[tag]
                    value2 = meta2[tag]
                    value3 = meta3[tag]
                    new_meta.addItem, tag, [value1, [value2], [value3]]
                  endif
                endelse
              endforeach
              
              if new_meta.hasTag('DATA IGNORE VALUE') then begin
                new_meta.updateItem, 'DATA IGNORE VALUE', 0.0
              endif else begin
                new_meta.addItem, 'DATA IGNORE VALUE', 0.0
              endelse
              
;              if new_meta.hasTag('SENSOR TYPE') then begin
;                new_meta.updateItem, 'SENSOR TYPE', 'ASTER'
;              endif else begin
;                new_meta.addItem, 'SENSOR TYPE', 'ASTER'
;              endelse
;              
;              new_meta.addItem, 'CLOUD COVER', meta1['CLOUD COVER']
;              new_meta.addItem, 'SUN AZIMUTH', meta1['SUN AZIMUTH']
;              new_meta.addItem, 'SUN ELEVATION', meta1['SUN ELEVATION']
              
              
              stack = ENVIRaster(temp.getData(), URI=stack_fn, metadata=new_meta, spatialRef=geo, interleave='BSQ')
              stack.save
              
              VNIR.close
              SWIR.close
              TIR.close
              temp.close
              stack.close
              
              end_time = systime(/seconds)
              seconds = end_time - start_time
              hours = string(fix(seconds / 3600.0))
              minutes = string(fix((seconds - (hours * 3600.0)) / 60.0))
              seconds = string(fix(seconds - (hours * 3600.0) - (minutes * 60.0)))
              print, strcompress('Consume:  ' + hours + 'h' + minutes + 'm' + seconds + 's')
  
  
            endelse
  
          endif else begin
  
            print, 'False Data!'
            false_list = [false_list, [basename]]
  
          endelse
  
        endelse
        
      endif else begin
        
        print, 'File does not exist!'
        no_exist_list = [no_exist_list, [basename]]
        
      endelse

  endfor
  
  file_delete, temp_dir, /recursive
  
  if n_elements(false_list) gt 0 then begin
    print, 'false list:'
    forEach basename, false_list do print, basename
  endif
  
  if n_elements(false_list) gt 0 then begin
    print, 'no exist list:'
    forEach basename, no_exist_list do print, basename
  endif

end

function get_fn_list, root_dir, csv_fn

  if strlen(csv_fn) gt 0 then begin
    _list = read_csv(csv_fn, count=count)
    _list = _list.field1
    fn_list = []
    forEach fn, _list do begin
      fn = strjoin([root_dir, '\', fn, '.hdf'], '')
      fn_list = [fn_list, [fn]]
    endforeach
    
  endif else begin
    fn_list = file_search(root_dir, '*.hdf')
  endelse
  
  return, fn_list
  
end

function get_ext, fn

  return, '.' + (strsplit(fn, '.', /extract))[-1]

end

function Layer_Stack, INPUT_RASTERS, OUTPUT_RASTER_URI

  e = ENVI()
  if file_test(OUTPUT_RASTER_URI) then begin
    return, e.openRaster(OUTPUT_RASTER_URI)
  endif else begin
    ;初始化Task
    Task = ENVITask('BuildLayerStack')
    ;设置输入参数
    Task.INPUT_RASTERS = INPUT_RASTERS
;    Task.GRID_DEFINITION = 'Coord. system + extents + pixel size'
    Task.RESAMPLING = 'Nearest Neighbor'
    Task.OUTPUT_RASTER_URI = OUTPUT_RASTER_URI
    ;执行Task
    Task.Execute
    ;获取输出结果
    return, Task.OUTPUT_RASTER
  endelse

end