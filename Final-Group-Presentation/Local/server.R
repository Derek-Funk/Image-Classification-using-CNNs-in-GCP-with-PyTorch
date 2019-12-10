server = function(input, output) {
  hide(id = "image1")
  hide(id = "image2")
  
  temp = reactiveValues(
    value = 2
  )
  
  observeEvent(eventExpr = input$uploadImage, handlerExpr = {
    # print(input$uploadImage$name)
    if(input$uploadImage$name == "12883_idx5_x1951_y551_class0.png") {
      temp$value = 0
    } else if(input$uploadImage$name == "12908_idx5_x2001_y1351_class1.png") {
      temp$value = 1
    } else {
      temp$value = NULL
    }
    
    print(temp$value)
    
    if(temp$value == 0) {
      hide("image2")
      show("image1")
    } else if(temp$value == 1) {
      hide("image1")
      show("image2")
    } else {
      NULL
    }
  })
  
  output$prediction = renderUI(expr = {
    if(temp$value == 0) {
      x = HTML("<em>Prediction: non-IDC</em>")
    } else if(temp$value == 1) {
      x = HTML("<em>Prediction: IDC</em>")
    } else {
      x = HTML("<em>Waiting for Upload...</em>")
    }
    x
  })
  
  # output$image = renderImage(expr = {
  #   if(temp$value == 0) {
  #     tags$img(
  #       src = "12883_idx5_x1951_y551_class0.png",
  #       height = "60%",
  #       width = "60%"
  #     )
  #   } else if(temp$value == 1) {
  #     tags$img(
  #       src = "12908_idx5_x2001_y1351_class1.png",
  #       height = "60%",
  #       width = "60%"
  #     )
  #   } else {
  #     NULL
  #   }
  # })
  
}