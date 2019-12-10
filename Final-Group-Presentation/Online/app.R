#library to deploy app: rsconnect
#command to deploy app: deployApp("C:\\Users\\derek.funk\\Desktop\\MSDS\\2019 Fall\\ML II\\Final Project\Final-Group-Presentation\\Online")

source("global.R")

ui = navbarPage(title = "Cancer Image Classification", theme = shinytheme(theme = "darkly"), useShinyjs(),
  tabPanel(title = "Background",
    includeHTML("www/background.html")         
  ),
  tabPanel(title = "Methods",
    includeHTML("www/methods.html")         
  ),
  tabPanel(title = "Data",
    box(width = 6,
      includeHTML("www/data.html")
    ),
    box(width = 6,
        h4("Example of Benign Image"),
        tags$img(
          src = "12883_idx5_x1951_y551_class0.png",
          height = "60%",
          width = "60%"
        ),
        br(), br(), h4("Example of Malignant Image"),
        tags$img(
          src = "12908_idx5_x2001_y1351_class1.png",
          height = "60%",
          width = "60%"
        )
    )
  ),
  tabPanel(title = "Models",
    includeHTML("www/models.html")
  ),
  tabPanel(title = "Upload Example",
    fileInput(inputId = "uploadImage", label = "Upload Image", accept = c('image/png', 'image/jpeg'))
    ,
    # conditionalPanel(
    #   condition = "input.uploadImage['name'] == '12883_idx5_x1951_y551_class0.png'",
    #   "asdf"
    # )
    # imageOutput(outputId = "image")
    column(width = 6,
      box(id = "image1", width = 12,
        tags$img(
          src = "12883_idx5_x1951_y551_class0.png",
          height = "60%",
          width = "60%"
        )
      ),
      box(id = "image2", width = 12,
          tags$img(
            src = "12908_idx5_x2001_y1351_class1.png",
            height = "60%",
            width = "60%"
          )
      ),
      uiOutput(outputId = "prediction")
    )
  ),
  tabPanel(title = "Future Improvements",
    includeHTML("www/future.html")
  )
)

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

shinyApp(ui, server)