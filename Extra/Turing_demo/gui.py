

from connect import *

###################################
##### Create a Simple GUI to allow a user to input the prefix for a ROI
###################################
import clr
clr.AddReference("System.Windows.Forms")
clr.AddReference("System.Drawing")

from System.Windows.Forms import Application, Form, Label, ComboBox, Button, TextBox
from System.Drawing import Point, Size

# Define Forms class that will prompt the user to select an
# ROI for creating isocenter data for the current beam set
class SelectROIForm(Form):
  def __init__(self, plan):
    # Set the size of the form
    self.Size = Size(500, 200)
    # Set title of the form
    self.Text = 'Approximate fiducial location'

    # Add a label
    label = Label()
    label.Text = 'Type the approximate fiducial location in mm'
    label.Location = Point(15, 15)
    label.AutoSize = True
    self.Controls.Add(label)

    # Add a TextBox
    self.textbox = TextBox()
    self.textbox.Location = Point(15,60)
    self.textbox.Width = 150
    self.Controls.Add(self.textbox)

    # Add button to press OK and close the form
    button = Button()
    button.Text = 'OK'
    button.AutoSize = True
    button.Location = Point(15, 100)
    button.Click += self.ok_button_clicked
    self.Controls.Add(button)
  
  def ok_button_clicked(self, sender, event):
    # Method invoked when the button is clicked
    # Save the selected ROI name
    self.roi_prefix = self.textbox.Text
    #self.roi_name = self.combobox.SelectedValue
    # Close the form
    self.Close()

# Access current plan and show the form
plan = get_current('Plan')

# Create an instance of the form and run it
form = SelectROIForm(plan)
Application.Run(form)

users_roi_prefix_label = form.roi_prefix




