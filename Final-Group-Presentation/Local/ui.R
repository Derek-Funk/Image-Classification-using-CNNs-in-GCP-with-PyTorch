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