
ACTION_CORR_LIST =  [
        {
            "name": "click",
            "description": "Click on an element with coordinates on the screenshot of the webpage.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "coords": {
                        "type": "list",
                        "description": "The coordinates of the element in the image to click: [x,y]"
                    }
                },
                "required": ["coords"]
            }
        },
        {
            "name": "type",
            "description": "Type content into a field with a specific id",
            "parameters": {
                "type": "object",
                "properties": {
                    "coords": {
                        "type": "list",
                        "description": "The coordinates of the element in the image to click: [x,y]"
                    },
                    "content": {
                        "type": "string",
                        "description": "Text to be typed"
                    },
                    "press_enter_after": {
                        "type": "integer",
                        "description": "Whether to press Enter after typing (1 by default, 0 to disable)",
                        "default": 0
                    }
                },
                "required": ["coords", "content"]
            }
        },
        {
            "name": "hover",
            "description": "Hover over an element with the coordinates",
            "parameters": {
                "type": "object",
                "properties": {
                    "coords": {
                        "type": "list",
                        "description": "The coordinates of the element in the image to hover: [x,y]"
                    }
                },
                "required": ["coords"]
            }
        },
        {
            "name": "press",
            "description": "Simulate pressing a key or a key combination",
            "parameters": {
                "type": "object",
                "properties": {
                    "coords": {
                        "type": "list",
                        "description": "The coordinates of the element in the image to hover: [x,y]"
                    },
                    "key": {
                        "type": "string",
                        "description": "a key or a key combination to press (e.g., 'ctrl+v' or 'enter')"
                    }
                },
                "required": ["key"]
            }
        },
        {
            "name": "scroll",
            "description": "Scroll the page up or down",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "Direction to scroll"
                    },
                    "distance": {
                        "type": "integer",
                        "description": "The scroll distance"
                    }
                },
                "required": ["direction", "distance"]
            }
        },
        {
            "name": "hscroll",
            "description": "Scroll the page horizontally",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Direction to scroll horizontally"
                    },
                    "distance": {
                        "type": "integer",
                        "description": "The scroll distance"
                    }
                },
                "required": ["direction", "distance"]
            }
        },
        {
            "name": "new_tab",
            "description": "Open a new, empty browser tab",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "tab_focus",
            "description": "Switch browser focus to a specific tab",
            "parameters": {
                "type": "object",
                "properties": {
                    "tab_index": {
                        "type": "integer",
                        "description": "Index of the tab to focus"
                    }
                },
                "required": ["tab_index"]
            }
        },
        # {
        #     "name": "close_tab",
        #     "description": "Close the currently active browser tab",
        #     "parameters": {
        #         "type": "object",
        #         "properties": {}
        #     }
        # },
        {
            "name": "go_back",
            "description": "Navigate to the previously viewed page",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "go_forward",
            "description": "Navigate to the next page after a 'go_back' action",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "move_to",
            "description": "Move the cursor to a specific location without clicking",
            "parameters": {
                "type": "object",
                "properties": {
                    "coords": {
                        "type": "list",
                        "description": "The coordinates to move the cursor to: [x,y]"
                    }
                },
                "required": ["coords"]
            }
        },
        {
            "name": "double_click",
            "description": "Perform a double click at a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "coords": {
                        "type": "list",
                        "description": "The coordinates to double click: [x,y]"
                    }
                },
                "required": ["coords"]
            }
        },
        {
            "name": "goto",
            "description": "Navigate to a specific URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to navigate to"
                    }
                },
                "required": ["url"]
            }
        },
        {
            "name": "browser_select_option",
            "description": "Select an option from a dropdown menu before browser_select_option you need to click the dropdown menu first",
            "parameters": {
                "type": "object",
                "properties": {
                    "coords": {
                        "type": "list",
                        "description": "The coordinates of the dropdown menu: [x,y]"
                    },
                    "option": {
                        "type": "string",
                        "description": "The option to select"
                    }
                },
                "required": ["coords", "option"]
            }
        },  
      {
          "name": "wait", 
          "description": "Wait for the change to happen", 
          "parameters": {
              "type": "object", 
              "properties": {
                  "seconds": {
                      "type": "integer", 
                      "description": "The seconds to wait"
                  }
              }, 
              "required": ["seconds"]
          }
      },
]


        # {
        #     "name": "new_tab",
        #     "description": "Open a new, empty browser tab",
        #     "parameters": {
        #         "type": "object",
        #         "properties": {}
        #     }
        # },
        # {
        #     "name": "tab_focus",
        #     "description": "Switch browser focus to a specific tab",
        #     "parameters": {
        #         "type": "object",
        #         "properties": {
        #             "tab_index": {
        #                 "type": "integer",
        #                 "description": "Index of the tab to focus"
        #             }
        #         },
        #         "required": ["tab_index"]
        #     }
        # },


ACTION_ID_LIST = [
        {
            "name": "click",
            "description": "Click on an element with a specific id on the webpage",
            "parameters": {
                "type": "object", 
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The id refers to the nearest number displayed on the image to the control element."
                    }
                },
                "required": ["id"]
            }
        },
        {
            "name": "type",
            "description": "Type content into a field with a specific id",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer", 
                        "description": "The id refers to the nearest number displayed on the image to the control field."
                    },
                    "content": {
                        "type": "string",
                        "description": "Text to be typed"
                    },
                    "press_enter_after": {
                        "type": "integer",
                        "description": "Whether to press Enter after typing (1 by default, 0 to disable)",
                        "default": 0
                    }
                },
                "required": ["id", "content"]
            }
        },
        {
            "name": "hover",
            "description": "Hover over an element with a specific id",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The id refers to the nearest number displayed on the image to the control element."
                    }
                },
                "required": ["id"]
            }
        },
        {
            "name": "press",
            "description": "Simulate pressing a key combination",
            "parameters": {
                "type": "object",
                "properties": {
                    "key_combination": {
                        "type": "string",
                        "description": "Key combination to press (e.g., 'Ctrl+v')"
                    }
                },
                "required": ["key_combination"]
            }
        },
        {
            "name": "scroll",
            "description": "Scroll the page up or down",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "Direction to scroll"
                    }
                },
                "required": ["direction"]
            }
        },

        # {
        #     "name": "close_tab",
        #     "description": "Close the currently active browser tab",
        #     "parameters": {
        #         "type": "object",
        #         "properties": {}
        #     }
        # },

        {
            "name": "go_back",
            "description": "Navigate to the previously viewed page",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "go_forward",
            "description": "Navigate to the next page after a 'go_back' action",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        },
        # {
        #     "name": "stop",
        #     "description": "The user's request has been completed. If the user needs you to answer any related questions, please provide the corresponding answers.",
        #     "parameters": {
        #         "type": "object", 
        #         "properties": {
        #             "answer": {
        #                 "type": "str",
        #                 "description": "The corresponding answers for the user's request."
        #             }
        #         },
        #     }
        # },
]