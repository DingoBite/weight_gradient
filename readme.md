# Stable Diffusion web-ui from automatic1111 extension. Weight Gradient

To use the extension, install it through the "install from URL" tab. Paste url of this repo

Next, enable the extension in the newly appeared tab. Now you have a new way of assigning weights to tokens. This method involves dynamically changing the weights of a set of tokens. In general, the format looks like this:

{ **tokens** : *start_step* - *end_step* : **start_weight** - **end_weight** - *return_weight* : *gradient_mode* }

**tokens** - the words to which the specified weight should be applied.

*start_step*, *end_step* - the initial and final steps of the process. The gradient starts at *start_step* and ends at *end_step*. The weight is retained after the last step. **The step format can be in percentages.** *OPTIONAL: no params - process runs throughout the generation.*

**start_weight**, **end_weight** - the initial and final weight of the process. The weight is set for each step between *start_step* and *end_step* according to the *gradient_mode*.

*return_weight* - the weight to which the process returns after **end_weight**. *OPTIONAL: if specified, end_weight is moved to the middle of the process. See examples for clarify.*

*gradient_mode* - the formula of weight change during the gradient process. It can be one of: a, e, ei, eo, c, ci, co.
*OPTIONAL: If no parameter is specified, the mode is linear (a). See examples for clarify.*

***You can use many of these constructions in one prompt***

***After process weight save the end weight. For example, { dog : 1 - 10 : 0 - 1 } -> .... [dog : 1]***

There are illustrations of the gradient modes in the "Modes Hints" tab of extension.


# Examples:

## 1. {dog : 0.1 - 0.3 : 1 - 0} 

Tokens: "dog ". Segment: 2 - 6.  Weights: 1.0 - 0.0

1.0 | 0.75 | 0.5 | 0.25 | 0.0

\* generation steps is 22

\* Extends to: 

([[dog : 1] :: 2]:1.0)([[dog : 2] :: 3]:0.75)([[dog : 3] :: 4]:0.5)([[dog : 4] :: 5]:0.25)

## 2. {cat : 1 - 15 : 1 - 0 - 1 : e}

Dynamic mode: e. Tokens: "cat ". Segment: 1 - 15.  Weights: 1.0 - 0.0 - 1.0

1 | 1.0 | 0.97 | 0.81 | 0.19 | 0.03 | 0.0 | 0.0 | 0.0 | 0.03 | 0.19 | 0.81 | 0.97 | 1.0 | 1

\* Extends to: 

([cat :: 1] : 1)([[cat : 1] :: 2]:0.9965)([[cat : 2] :: 3]:0.9744)([[cat : 3] :: 4]:0.8143)([[cat : 4] :: 5]:0.1857)([[cat : 5] :: 6]:0.0256)([[cat : 6] :: 7]:0.0035)([[cat : 8] :: 9]:0.0035)([[cat : 9] :: 10]:0.0256)([[cat : 10] :: 11]:0.1857)([[cat : 11] :: 12]:0.8143)([[cat : 12] :: 13]:0.9744)([[cat : 13] :: 14]:0.9965)([cat : 15]:1)

## 3. {redhead girl, drinking coffee :: 0 - 1 : c}

Dynamic mode: c. Tokens: "redhead girl, drinking coffee ". Segment: 1 - 22.  Weights: 0.0 - 1.0

0.0 | 0.0 | 0.01 | 0.02 | 0.04 | 0.06 | 0.09 | 0.13 | 0.18 | 0.24 | 0.35 | 0.65 | 0.76 | 0.82 | 0.87 | 0.91 | 0.94 | 0.96 | 0.98 | 0.99 | 1.0 | 1

\* Extends to:

([[redhead girl, drinking coffee : 1] :: 2]:0.0023)([[redhead girl, drinking coffee : 2] :: 3]:0.0092)([[redhead girl, drinking coffee : 3] :: 4]:0.0208)([[redhead girl, drinking coffee : 4] :: 5]:0.0377)([[redhead girl, drinking coffee : 5] :: 6]:0.0603)([[redhead girl, drinking coffee : 6] :: 7]:0.0897)([[redhead girl, drinking coffee : 7] :: 8]:0.1273)([[redhead girl, drinking coffee : 8] :: 9]:0.1762)([[redhead girl, drinking coffee : 9] :: 10]:0.2425)([[redhead girl, drinking coffee : 10] :: 11]:0.3475)([[redhead girl, drinking coffee : 11] :: 12]:0.6525)([[redhead girl, drinking coffee : 12] :: 13]:0.7575)([[redhead girl, drinking coffee : 13] :: 14]:0.8238)([[redhead girl, drinking coffee : 14] :: 15]:0.8727)([[redhead girl, drinking coffee : 15] :: 16]:0.9103)([[redhead girl, drinking coffee : 16] :: 17]:0.9397)([[redhead girl, drinking coffee : 17] :: 18]:0.9623)([[redhead girl, drinking coffee : 18] :: 19]:0.9792)([[redhead girl, drinking coffee : 19] :: 20]:0.9908)([[redhead girl, drinking coffee : 20] :: 21]:0.9977)([redhead girl, drinking coffee : 22]:1)

# Details:
### 1. Weight accuracy - 4 numbers after the decimal point
### 2. Steps rounding is to floor. 10.5 -> 10



# RU
# Stable Diffusion web-ui from automatic1111 extension. Weight Gradient


Чтобы использовать расширение, установите его через вкладку "install from URL". Вставьте URL этого репозитория.

Затем включите расширение во вновь появившейся вкладке в режиме text2img или img2img. Теперь у вас есть новый способ назначения весов токенам. Этот метод включает в себя динамическое изменение весов токенов. В общем, формат выглядит так:

{ **tokens** : *start_step* - *end_step* : **start_weight** - **end_weight** - *return_weight* : *gradient_mode* }

**tokens** - слова, к которым должен быть применен указанный вес.

*start_step*, *end_step* - начальный и конечный шаги процесса. Градиент начинается на *start_step* и заканчивается на *end_step*. Вес сохраняется после последнего шага. **Формат шага может быть в процентах**. *ОПЦИОНАЛЬНО: без параметров - процесс запускается в течение всего поколения.*

**start_weight**, **end_weight** - начальный и конечный вес процесса. Вес устанавливается для каждого шага между *start_step* и *end_step* в соответствии с *gradient_mode*.

*return_weight* - вес, к которому вернется значение после **end_weight**. *ОПЦИОНАЛЬНО: если указано, то конечный вес перемещается в середину процесса. Смотрите примеры для уточнения.*

*gradient_mode* - формула изменения веса в процессе градиента. Она может быть одной из:: a, e, ei, eo, c, ci, co.
*ОПЦИОНАЛЬНО: Если параметр не указан, то режим линейный (a). Смотрите примеры для уточнения.*

***Вы можете использовать множество конструкций в одном промпте***

Иллюстрации режимов градиента можно найти на вкладке "Modes Hints" расширения.
